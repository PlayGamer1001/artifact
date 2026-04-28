# -*- coding: utf-8 -*-
"""Parse HTML/MD/TXT files to sentence-level JSONL with language detection and table processing."""

from __future__ import annotations

import codecs
import concurrent.futures
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import tiktoken
except ImportError as exc:
    raise RuntimeError("tiktoken is required for table token rules, but is not installed.") from exc
try:
    from lingua import Language, LanguageDetectorBuilder
except ImportError as exc:
    raise RuntimeError("lingua-language-detector is required for language filtering.") from exc
try:
    from bs4 import BeautifulSoup, NavigableString, Tag
except ImportError as exc:
    raise RuntimeError("beautifulsoup4 is required for html-table detection in markdown stage.") from exc

from .convert import convert_markitdown_file, convert_html_table_fragment


INLINE_LINK_RE = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
HEADER_RE = re.compile(r'^\s{0,3}(#{1,6})\s+(.*)$')
LIST_RE = re.compile(r'^\s{0,3}(?:[-*+]|\d+[\.)]|[A-Za-z][\.)])\s+(.*)$')
TABLE_SEP_RE = re.compile(r'^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$')
INLINE_LIST_MARKER_RE = re.compile(
    r'(?:(?<=^)|(?<=\s))(?P<marker>(?:[-*+•·●])|(?:\d+[\.)])|(?:[A-Za-z][\.)]))(?=\s+)'
)
TABLE_TOKEN_THRESHOLD = 512
try:
    TABLE_TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception as exc:
    raise RuntimeError(
        "Failed to initialize tiktoken encoding 'cl100k_base'. "
        "Package may be installed, but encoding data could not be loaded "
        "(often proxy/network/cache issue)."
    ) from exc

LANG_DETECTOR = LanguageDetectorBuilder.from_all_languages().build()
MD_SUFFIXES = {'.md'}
TXT_SUFFIXES = {'.txt'}
HTML_SUFFIXES = {'.html', '.htm'}
HTML_PARSER = "html5lib"
WHITE_SPACE_STYLE_RE = re.compile(r"white-space\s*:\s*(pre|pre-line|pre-wrap)", re.IGNORECASE)


def _normalize_unicode_text(text):
    value = str(text or '')
    value = unicodedata.normalize('NFKC', value)
    value = value.replace('\xa0', ' ')
    return re.sub(r'\s+', ' ', value).strip()


def _materialize_newlines_as_p(soup):
    for node in soup.find_all(True):
        style = str(node.get("style", ""))
        if str(node.name).lower() != "pre" and WHITE_SPACE_STYLE_RE.search(style) is None:
            continue
        for text_node in list(node.find_all(string=True)):
            if not isinstance(text_node, NavigableString):
                continue
            text = str(text_node)
            if "\n" not in text:
                continue
            parts = [p for p in text.split("\n") if len(str(p).strip()) > 0]
            if len(parts) == 0:
                continue
            first_p = soup.new_tag("p")
            first_p.append(parts[0])
            text_node.replace_with(first_p)
            cursor = first_p
            for part in parts[1:]:
                nxt_p = soup.new_tag("p")
                nxt_p.append(part)
                cursor.insert_after(nxt_p)
                cursor = nxt_p


def _marker_style(marker):
    value = str(marker or '').strip()
    if re.fullmatch(r'[-*+•·]', value):
        return f'bullet:{value}'
    if re.fullmatch(r'\d+\.', value):
        return 'digit_dot'
    if re.fullmatch(r'\d+\)', value):
        return 'digit_paren'
    if re.fullmatch(r'[A-Za-z]\.', value):
        return 'alpha_dot'
    if re.fullmatch(r'[A-Za-z]\)', value):
        return 'alpha_paren'
    return None


def _split_inline_same_bullets_to_ul(soup):
    block_tags = {"p", "div", "section", "article", "td", "th"}
    structural_descendants = [
        "h1", "h2", "h3", "h4", "h5", "h6", "p", "div", "section", "article", "ul", "ol", "li", "table"
    ]
    for node in soup.find_all(block_tags):
        if node.find("br") is not None:
            continue
        if node.find_parent(["ul", "ol", "table"]):
            continue
        if node.find(structural_descendants) is not None:
            continue
        raw = node.get_text(" ", strip=True)
        if not raw:
            continue
        markers = list(INLINE_LIST_MARKER_RE.finditer(raw))
        if len(markers) < 2:
            continue
        styles = {_marker_style(m.group("marker")) for m in markers}
        if None in styles or len(styles) != 1:
            continue
        parts = INLINE_LIST_MARKER_RE.split(raw)
        segs = []
        for i, seg in enumerate(parts):
            if i % 2 == 1:
                continue
            seg = re.sub(r"\s+", " ", str(seg)).strip()
            if seg:
                segs.append(seg)
        if len(segs) < 3:
            continue

        prefix = segs[0]
        items = segs[1:]
        node.clear()
        if prefix:
            p = soup.new_tag("p")
            p.append(prefix)
            node.append(p)
        ul = soup.new_tag("ul")
        for item in items:
            li = soup.new_tag("li")
            li.append(item)
            ul.append(li)
        node.append(ul)


def preprocess_html_before_parse(raw_html):
    soup = BeautifulSoup(raw_html, HTML_PARSER)
    _materialize_newlines_as_p(soup)
    _split_inline_same_bullets_to_ul(soup)
    return str(soup)


def _contains_letters_or_numbers(text):
    return bool(re.search(r'[A-Za-z0-9]', str(text or '')))


def _strip_markdown_inline_links(text):
    return INLINE_LINK_RE.sub(r'\1', str(text))


def _split_table_cells(line):
    s = line.strip()
    if not s:
        return []
    if s.startswith('|'):
        s = s[1:]
    if s.endswith('|'):
        s = s[:-1]
    cells = [re.sub(r'\s+', ' ', c).strip() for c in s.split('|')]
    return [c for c in cells if c]


def _looks_like_markdown_table_row(raw_line):
    s = str(raw_line or '').strip()
    if len(s) == 0:
        return False
    if TABLE_SEP_RE.match(s):
        return True
    if '|' not in s:
        return False
    if not (s.startswith('|') or s.endswith('|')):
        return False
    return len(_split_table_cells(s)) >= 1


def _looks_like_html_table_line(raw_line):
    s = str(raw_line or '').lower()
    return ('<table' in s) or ('<tr' in s) or ('<td' in s) or ('<th' in s)


def _normalize_multiline_cell_text(text):
    lines = [re.sub(r'\s+', ' ', ln).strip() for ln in str(text or '').splitlines()]
    lines = [ln for ln in lines if len(ln) > 0]
    return '\n'.join(lines)


def _extract_html_table_rows(block_lines):
    fragment = "\n".join([str(x) for x in block_lines])
    soup = BeautifulSoup(fragment, HTML_PARSER)
    rows = []

    for tr in soup.find_all('tr'):
        cells = []
        for cell in tr.find_all(['th', 'td']):
            txt = _normalize_multiline_cell_text(cell.get_text('\n', strip=True))
            if len(txt) > 0:
                cells.append(txt)
        if len(cells) > 0:
            rows.append(cells)

    if len(rows) == 0:
        loose_cells = []
        for cell in soup.find_all(['th', 'td']):
            txt = _normalize_multiline_cell_text(cell.get_text('\n', strip=True))
            if len(txt) > 0:
                loose_cells.append(txt)
        if len(loose_cells) > 0:
            rows.append(loose_cells)
    return rows


def _token_count(text):
    text = str(text or '')
    return len(TABLE_TOKEN_ENCODER.encode(text))


def _detect_language_code(text):
    value = str(text or '')
    if len(value) < 20:
        return "unknown"

    lang = LANG_DETECTOR.detect_language_of(value)
    if lang is None:
        return "unknown"
    if lang == Language.ENGLISH:
        return "en"

    iso = getattr(lang, "iso_code_639_1", None)
    if iso is not None:
        return str(getattr(iso, "name", iso)).lower()
    return str(getattr(lang, "name", "unknown")).lower()


def _classify_table_row_rule(cells, table_rows, table_cols, threshold=TABLE_TOKEN_THRESHOLD):
    non_empty_cells = [str(c) for c in cells if len(str(c).strip()) > 0]
    if len(non_empty_cells) == 0:
        return ("empty", [])
    tokens = [_token_count(c) for c in non_empty_cells]
    has_oversized = any([t > threshold for t in tokens])
    all_non_empty_oversized = all([t > threshold for t in tokens])
    is_single_row_or_col = (table_rows == 1) or (table_cols == 1)
    if all_non_empty_oversized:
        return ("rule_b_all_non_empty_oversized", tokens)
    if is_single_row_or_col and has_oversized:
        return ("rule_a_single_row_or_col_with_oversized", tokens)
    return ("default_table", tokens)


def _normalize_inline_text(text):
    return re.sub(r'\s+', ' ', str(text or '')).strip()


def _is_plain_list_item_text(text):
    normalized = _normalize_inline_text(text)
    if len(normalized) == 0:
        return False
    if normalized.endswith(('.', '?', '!', ':', ';')):
        return False
    if ':' in normalized:
        return False
    if re.search(r'\s\/\s', normalized):
        return False
    return True


def _get_enumeration_style(text):
    normalized = _normalize_inline_text(text)
    if len(normalized) == 0:
        return None
    patterns = [
        ("bullet_symbol", r'^[\-*•·●+]\s+'),
        ("digit_dot", r'^\d+\.\s+'),
        ("digit_paren", r'^\d+\)\s+'),
        ("alpha_dot", r'^[A-Za-z]\.\s+'),
        ("roman_paren", r'^\([ivxIVX]+\)\s+'),
        ("alpha_paren", r'^\([A-Za-z]\)\s+'),
        ("roman_dot", r'^[ivxIVX]+\.\s+'),
    ]
    for style, pattern in patterns:
        if re.match(pattern, normalized):
            return style
    return None


def _inline_marker_style(marker):
    value = str(marker or '').strip()
    if re.fullmatch(r'[-*+•·●]', value):
        return f'bullet:{value}'
    if re.fullmatch(r'\d+\.', value):
        return 'digit_dot'
    if re.fullmatch(r'\d+\)', value):
        return 'digit_paren'
    if re.fullmatch(r'[A-Za-z]\.', value):
        return 'alpha_dot'
    if re.fullmatch(r'[A-Za-z]\)', value):
        return 'alpha_paren'
    return None


def _get_list_style_for_record(rec):
    source_type = rec.get('source_type', 'paragraph')
    sentence = _normalize_inline_text(rec.get('sentence', ''))
    if len(sentence) == 0:
        return None
    if source_type == 'list':
        return 'assoc_list'
    enum_style = _get_enumeration_style(sentence)
    if enum_style is not None:
        return enum_style
    return 'plain' if _is_plain_list_item_text(sentence) else None


def _compose_virtual_list_item(item_text, intro_text):
    intro = _normalize_inline_text(intro_text)
    if intro.endswith('.'):
        intro = intro[:-1].rstrip()
    if not intro.endswith(':'):
        intro = intro + ':'
    item = _normalize_inline_text(item_text)
    if len(item) == 0:
        return ''
    return f'{intro} {item}'.strip()


def _apply_synthetic_list_merge(records):
    merged = []
    i = 0
    while i < len(records):
        cur = records[i]
        sentence = str(cur.get('sentence', '')).strip()
        source_type = cur.get('source_type', 'paragraph')
        if source_type in ['paragraph', 'list'] and sentence.endswith(':'):
            j = i + 1
            items = []
            first_style = None
            while j < len(records):
                nxt = records[j]
                nxt_type = nxt.get('source_type', 'paragraph')
                if nxt_type not in ['paragraph', 'list']:
                    break
                nxt_sentence = str(nxt.get('sentence', '')).strip()
                if len(nxt_sentence) == 0:
                    break
                style = _get_list_style_for_record(nxt)
                if style is None:
                    break
                if first_style is None:
                    first_style = style
                elif style != first_style:
                    break
                items.append(nxt)
                j += 1

            if len(items) > 0:
                for item in items:
                    merged.append(
                        {
                            'section': cur.get('section', 'General'),
                            'sentence': _compose_virtual_list_item(item.get('sentence', ''), sentence),
                            'source_type': 'list',
                        }
                    )
                i = j
                continue

        merged.append(cur)
        i += 1
    return merged


def _split_inline_bullets_to_records(text, section):
    value = str(text or '')
    markers = list(INLINE_LIST_MARKER_RE.finditer(value))
    if len(markers) < 2:
        return None

    marker_styles = {_inline_marker_style(m.group('marker')) for m in markers}
    if None in marker_styles or len(marker_styles) != 1:
        return None

    parts = INLINE_LIST_MARKER_RE.split(value)
    if len(parts) < 3:
        return None

    prefix = _normalize_inline_text(parts[0])
    items = [_normalize_inline_text(p) for p in parts[1:] if len(_normalize_inline_text(p)) > 0]
    if len(items) < 2:
        return None

    recs = []
    if len(prefix) > 0:
        recs.append(
            {
                'section': section,
                'sentence': prefix,
                'source_type': 'paragraph',
            }
        )
    for item in items:
        recs.append(
            {
                'section': section,
                'sentence': item,
                'source_type': 'list',
            }
        )
    return recs


def _split_colon_sections_to_records(text, section):
    value = _normalize_inline_text(text)
    if len(value) < 300:
        return None

    marker_re = re.compile(r'(?<!https)(?<!http)(?<!www\.)\b([A-Z][A-Za-z0-9/&()\'\-\s]{2,80}:)')
    matches = list(marker_re.finditer(value))
    if len(matches) < 2:
        return None

    chunks = []
    start = 0
    for m in matches:
        idx = m.start()
        if idx > start:
            prev = _normalize_inline_text(value[start:idx])
            if prev:
                chunks.append(prev)
        start = idx
    tail = _normalize_inline_text(value[start:])
    if tail:
        chunks.append(tail)

    chunks = [c for c in chunks if len(c) > 0]
    if len(chunks) < 2:
        return None

    recs = []
    for c in chunks:
        recs.append(
            {
                'section': section,
                'sentence': c,
                'source_type': 'paragraph',
            }
        )
    return recs


def _split_dense_comma_list_to_records(text, section):
    value = _normalize_inline_text(text)
    if len(value) < 300:
        return None

    comma_count = value.count(',')
    period_count = value.count('.')
    semicolon_count = value.count(';')
    if comma_count < 20 or period_count > 4 or semicolon_count > 6:
        return None

    items = [_normalize_inline_text(x) for x in value.split(',')]
    items = [x for x in items if len(x) > 0]
    if len(items) < 15:
        return None

    avg_len = sum(len(x) for x in items) / max(1, len(items))
    if avg_len < 4 or avg_len > 50:
        return None

    recs = []
    for item in items:
        recs.append(
            {
                'section': section,
                'sentence': item,
                'source_type': 'list',
            }
        )
    return recs


def _load_markdown_text(filename):
    with open(filename, 'rb') as fp:
        raw = fp.read()
    for encoding in ['utf-8', 'utf-8-sig', 'latin-1']:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode('utf-8', errors='replace')


def _html_token_count(text):
    return len(TABLE_TOKEN_ENCODER.encode(str(text or '')))


def _to_markdown_table_row(cells):
    escaped = [str(c).replace("|", "\\|") for c in cells]
    return "| " + " | ".join(escaped) + " |"


def _html_text_segments(node):
    block_like = {
        "h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "table",
        "ul", "ol", "div", "section", "article", "details", "summary",
    }
    br_marker = "__HTML_BR_SPLIT__"
    parts = []

    def append_inline_text(cur):
        if isinstance(cur, NavigableString):
            parts.append(str(cur).replace('\n', ' '))
            return
        if not isinstance(cur, Tag):
            return
        cname = str(cur.name or '').lower()
        if cname == 'br':
            parts.append(br_marker)
            return
        if cname in block_like:
            return
        for sub in cur.children:
            append_inline_text(sub)

    for child in node.children:
        append_inline_text(child)

    lines = [_normalize_inline_text(x) for x in ''.join(parts).split(br_marker)]
    return [x for x in lines if x]


def _batch_convert_small_tables(records):
    pending = [(idx, rec) for idx, rec in enumerate(records) if rec.get('kind') == 'small_table_html']
    if len(pending) == 0:
        return records
    rec_idx_to_table_idx = {rec_idx: table_idx for table_idx, (rec_idx, _rec) in enumerate(pending)}

    chunks = ["<html><body>"]
    for table_idx, (_idx, rec) in enumerate(pending):
        marker = f"__TABLE_MARKER_{table_idx}__"
        chunks.append(f"<h6>{marker}</h6>")
        chunks.append(str(rec.get('html', '')))
    chunks.append("</body></html>")
    batch_html = "\n".join(chunks)

    parsed_by_idx = {}
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', encoding='utf-8', delete=True) as tmp:
            tmp.write(batch_html)
            tmp.flush()
            md = convert_markitdown_file(Path(tmp.name))
    except Exception:
        md = ''

    if md:
        marker_re = re.compile(r'__TABLE_MARKER_(\d+)__')
        current = None
        buf = []

        def flush_current():
            nonlocal current, buf
            if current is None:
                return
            parsed_by_idx[current] = '\n'.join(buf).strip()
            current = None
            buf = []

        for line in str(md).splitlines():
            m = marker_re.search(line)
            if m is not None:
                flush_current()
                current = int(m.group(1))
                continue
            if current is not None:
                buf.append(line)
        flush_current()

    out = []
    for idx, rec in enumerate(records):
        if rec.get('kind') != 'small_table_html':
            out.append(rec)
            continue

        table_idx = rec_idx_to_table_idx.get(idx)
        converted = ''
        if table_idx is not None:
            converted = parsed_by_idx.get(table_idx, '').strip()
        if not converted:
            try:
                converted = convert_html_table_fragment(str(rec.get('html', ''))).strip()
            except Exception:
                converted = ''

        section = rec.get('section', 'General')
        if converted:
            for ln in converted.splitlines():
                text = _normalize_inline_text(ln)
                if not text:
                    continue
                out.append(
                    {
                        'section': section,
                        'sentence': text,
                        'source_type': 'table',
                    }
                )
    return out


def _parse_html_records(raw_html):
    soup = BeautifulSoup(raw_html, HTML_PARSER)
    root = soup.body if soup.body is not None else soup
    records = []
    heading_stack = {}

    def current_section():
        if len(heading_stack) == 0:
            return 'General'
        parts = []
        for level in range(1, 7):
            title = heading_stack.get(level)
            if title:
                parts.append(f'H{level}:{title}')
        return ' > '.join(parts) if parts else 'General'

    def append_record(sentence, source_type):
        text = _normalize_inline_text(sentence)
        if not text:
            return
        records.append(
            {
                'section': current_section(),
                'sentence': text,
                'source_type': source_type,
            }
        )

    def walk(node, in_list=False):
        block_like = {
            "h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "table",
            "ul", "ol", "div", "section", "article", "details", "summary",
        }
        for child in node.children:
            if isinstance(child, NavigableString):
                # Some pages place policy body as direct text under body/div without <p>/<li>.
                # Treat those text nodes as virtual paragraphs and keep newline boundaries.
                raw_text = str(child)
                if not _contains_letters_or_numbers(raw_text):
                    continue

                # First pass: behave like a normal <p> node.
                virtual_p = soup.new_tag("p")
                virtual_p.append(raw_text)
                paragraph_like_segments = _html_text_segments(virtual_p)

                # Fallback: if it is still one huge segment, split by hard line breaks.
                if len(paragraph_like_segments) <= 1:
                    paragraph_like_segments = []
                    for part in re.split(r'(?:\r?\n)+', raw_text):
                        text_part = _normalize_inline_text(part)
                        if text_part and _contains_letters_or_numbers(text_part):
                            paragraph_like_segments.append(text_part)

                for seg in paragraph_like_segments:
                    append_record(seg, 'list' if in_list else 'paragraph')
                continue
            if not isinstance(child, Tag):
                continue
            name = str(child.name or '').lower()

            if name in {"div", "section", "article", "details", "ul", "ol"}:
                walk(child, in_list=in_list)
                continue

            if name == 'summary' or (name.startswith('h') and len(name) == 2 and name[1].isdigit()):
                level = 3 if name == 'summary' else int(name[1])
                header_text = _normalize_inline_text(child.get_text(' ', strip=True))
                if header_text:
                    heading_stack[level] = header_text
                    for lv in range(level + 1, 7):
                        heading_stack.pop(lv, None)
                continue

            if name == 'table':
                if child.find_parent('table') is not None:
                    continue
                for tr in child.find_all('tr'):
                    if tr.find_parent('table') is not child:
                        continue

                    cells = tr.find_all(['th', 'td'], recursive=False)
                    if len(cells) == 0:
                        continue

                    row_cells = []
                    row_cell_texts = []
                    for cell in cells:
                        cell_text = _normalize_inline_text(cell.get_text(' ', strip=True))
                        if not cell_text:
                            continue
                        row_cells.append(cell)
                        row_cell_texts.append(cell_text)

                    if len(row_cell_texts) == 0:
                        continue

                    row_md = _to_markdown_table_row(row_cell_texts)
                    row_tokens = _html_token_count(row_md)

                    if row_tokens < TABLE_TOKEN_THRESHOLD:
                        append_record(row_md, 'table')
                        continue

                    residual_cells = []
                    for idx, cell in enumerate(row_cells):
                        cell_text = row_cell_texts[idx]
                        if _html_token_count(cell_text) > TABLE_TOKEN_THRESHOLD:
                            walk(cell, in_list=False)
                            continue
                        residual_cells.append(cell_text)

                    if len(residual_cells) > 0:
                        append_record(_to_markdown_table_row(residual_cells), 'table')
                continue

            if name == 'li':
                for seg in _html_text_segments(child):
                    append_record(seg, 'list')
                nested_block_names = {"ul", "ol", "div", "section", "article", "details", "p", "li", "table"}
                for grand in child.children:
                    if not isinstance(grand, Tag):
                        continue
                    gname = str(grand.name or '').lower()
                    if gname in nested_block_names:
                        walk(grand, in_list=True)
                continue

            if name == 'p':
                for seg in _html_text_segments(child):
                    append_record(seg, 'list' if in_list else 'paragraph')
                if child.find(list(block_like)) is not None:
                    walk(child, in_list=in_list)
                continue

            has_block_descendant = child.find(list(block_like)) is not None
            if has_block_descendant:
                walk(child, in_list=in_list)
                continue

            for seg in _html_text_segments(child):
                append_record(seg, 'list' if in_list else 'paragraph')

    walk(root, in_list=False)
    records = _batch_convert_small_tables(records)
    records = _apply_synthetic_list_merge(records)
    return TextPostProcessor(records).post_process()


class TextPostProcessor:
    def __init__(self, document):
        self.document = document

    def post_process(self):
        res = []
        for rec in self.document:
            line = str(rec.get('sentence', ''))
            source_type = rec.get('source_type', 'paragraph')
            section = rec.get('section', 'General')

            if len(line.strip()) == 0:
                continue

            line = _normalize_unicode_text(line)
            line = _strip_markdown_inline_links(line)
            line = re.sub(r'\s+', ' ', line)
            line = re.sub(r'\s*\.', '.', line)
            line = re.sub(r'^\s*[\.,;:]\s*', '', line)
            line = re.sub(r':\(', ': (', line)
            line = re.sub(r'\)[A-Za-z0-9]', lambda m: m.group(0)[0] + ' ' + m.group(0)[1], line)
            line = re.sub(r'^\s*#+\s*', '', line)
            line = re.sub(r'^\s+', '', line)
            line = re.sub(r'\s+$', '', line)

            if len(line) == 0:
                continue

            if source_type != 'table' and not _contains_letters_or_numbers(line):
                continue

            res.append(
                {
                    'section': section,
                    'sentence': line,
                    'source_type': source_type,
                }
            )
        return res


class Preprocessor:
    def __init__(self, filename=None, mkdown_text=None):
        if mkdown_text is not None:
            self.mkdown = str(mkdown_text)
            return
        ext = os.path.splitext(str(filename))[1].lower()
        if ext != '.md':
            raise ValueError('Only .md input is supported in linux md-only mode: {}'.format(filename))
        self.mkdown = _load_markdown_text(filename)

    def parse(self):
        output = []
        paragraph_buffer = []
        in_code_block = False
        heading_stack = {}

        def current_section():
            if len(heading_stack) == 0:
                return 'General'
            parts = []
            for level in range(1, 7):
                title = heading_stack.get(level)
                if title:
                    parts.append(f'H{level}:{title}')
            return ' > '.join(parts) if parts else 'General'

        def flush_paragraph():
            if not paragraph_buffer:
                return
            text = re.sub(r'\s+', ' ', ' '.join(paragraph_buffer)).strip()
            paragraph_buffer.clear()
            if text:
                append_text_with_list_detection(text, default_source_type='paragraph')

        def append_table_record(sentence):
            output.append(
                {
                    'section': current_section(),
                    'sentence': sentence,
                    'source_type': 'table',
                }
            )

        def append_paragraph_record(sentence):
            output.append(
                {
                    'section': current_section(),
                    'sentence': sentence,
                    'source_type': 'paragraph',
                }
            )

        def append_list_record(sentence):
            output.append(
                {
                    'section': current_section(),
                    'sentence': sentence,
                    'source_type': 'list',
                }
            )

        def append_text_with_list_detection(text, default_source_type='paragraph'):
            raw = str(text or '')
            section = current_section()

            inline_split_records = _split_inline_bullets_to_records(raw, section)
            if inline_split_records is not None:
                output.extend(inline_split_records)
                return

            colon_split_records = _split_colon_sections_to_records(raw, section)
            if colon_split_records is not None:
                output.extend(colon_split_records)
                return

            comma_list_records = _split_dense_comma_list_to_records(raw, section)
            if comma_list_records is not None:
                output.extend(comma_list_records)
                return

            for part in str(raw).split('\n'):
                piece = part.strip()
                if len(piece) == 0:
                    continue
                m = LIST_RE.match(piece)
                if m:
                    item_text = m.group(1).strip()
                    if len(item_text) > 0:
                        append_list_record(item_text)
                    continue

                if default_source_type == 'table':
                    append_table_record(piece)
                else:
                    append_paragraph_record(piece)

        def process_fragment_text(fragment_text, depth=0, max_depth=6):
            if depth > max_depth:
                append_paragraph_record(_normalize_inline_text(fragment_text))
                return

            frag_lines = str(fragment_text or '').splitlines()
            local_buf = []
            local_in_code = False
            local_code_buf = []

            def flush_local_paragraph():
                if not local_buf:
                    return
                text = re.sub(r'\s+', ' ', ' '.join(local_buf)).strip()
                local_buf.clear()
                if text:
                    append_text_with_list_detection(text, default_source_type='paragraph')

            j = 0
            while j < len(frag_lines):
                raw = frag_lines[j]
                stripped_local = str(raw).strip()

                if stripped_local.startswith('```'):
                    if local_in_code:
                        process_code_block_lines(local_code_buf, depth=depth, max_depth=max_depth)
                        local_code_buf.clear()
                        local_in_code = False
                    else:
                        flush_local_paragraph()
                        local_in_code = True
                    j += 1
                    continue

                if local_in_code:
                    if stripped_local:
                        local_code_buf.append(stripped_local)
                    j += 1
                    continue

                if len(stripped_local) == 0:
                    flush_local_paragraph()
                    j += 1
                    continue

                if _looks_like_markdown_table_row(stripped_local) or _looks_like_html_table_line(stripped_local):
                    flush_local_paragraph()
                    nested_block = []
                    while j < len(frag_lines):
                        cur = str(frag_lines[j]).strip()
                        if len(cur) == 0:
                            break
                        if cur.startswith('```'):
                            break
                        if LIST_RE.match(cur) and not (_looks_like_markdown_table_row(cur) or _looks_like_html_table_line(cur)):
                            break
                        if _looks_like_markdown_table_row(cur) or _looks_like_html_table_line(cur):
                            nested_block.append(cur)
                            j += 1
                            continue
                        break
                    process_table_block(nested_block, depth=depth + 1, max_depth=max_depth)
                    continue

                m = LIST_RE.match(stripped_local)
                if m:
                    flush_local_paragraph()
                    item_text = m.group(1).strip()
                    if item_text:
                        append_list_record(item_text)
                    j += 1
                    continue

                local_buf.append(stripped_local)
                j += 1

            flush_local_paragraph()
            if local_in_code and len(local_code_buf) > 0:
                process_code_block_lines(local_code_buf, depth=depth, max_depth=max_depth)

        def is_table_candidate(raw_line):
            return _looks_like_markdown_table_row(raw_line) or _looks_like_html_table_line(raw_line)

        def handle_oversized_cells(cells, tokens, depth=0, max_depth=6):
            for idx, cell_text in enumerate(cells):
                cell_value = str(cell_text or '')
                is_oversized = idx < len(tokens) and tokens[idx] > TABLE_TOKEN_THRESHOLD
                if is_oversized:
                    process_fragment_text(cell_value, depth=depth + 1, max_depth=max_depth)
                else:
                    append_text_with_list_detection(cell_value, default_source_type='paragraph')

        def process_code_block_lines(code_lines, depth=0, max_depth=6):
            normalized = [str(x).strip() for x in code_lines if len(str(x).strip()) > 0]
            if len(normalized) == 0:
                return

            if any([is_table_candidate(x) for x in normalized]):
                process_table_block(normalized, depth=depth, max_depth=max_depth)
                return

            table_rows = len(normalized)
            table_cols = 1
            for row_text in normalized:
                cells = [row_text]
                rule_name, tokens = _classify_table_row_rule(
                    cells,
                    table_rows,
                    table_cols,
                    threshold=TABLE_TOKEN_THRESHOLD,
                )
                if rule_name in ['rule_b_all_non_empty_oversized', 'rule_a_single_row_or_col_with_oversized']:
                    handle_oversized_cells(cells, tokens, depth=depth, max_depth=max_depth)
                    continue
                append_table_record(row_text)

        def process_table_block(block_lines, depth=0, max_depth=6):
            has_html_table = any([_looks_like_html_table_line(x) for x in block_lines])
            if has_html_table:
                rows = _extract_html_table_rows(block_lines)
                table_rows = len(rows)
                table_cols = max([len(r) for r in rows], default=0)

                for cells in rows:
                    if len(cells) == 0:
                        continue
                    rule_name, tokens = _classify_table_row_rule(
                        cells,
                        table_rows,
                        table_cols,
                        threshold=TABLE_TOKEN_THRESHOLD,
                    )
                    if rule_name == 'rule_b_all_non_empty_oversized':
                        handle_oversized_cells(cells, tokens, depth=depth, max_depth=max_depth)
                        continue

                    if rule_name == 'rule_a_single_row_or_col_with_oversized':
                        handle_oversized_cells(cells, tokens, depth=depth, max_depth=max_depth)
                        continue

                    append_table_record(' | '.join(cells))
                return

            data_rows = []
            for row_line in block_lines:
                if TABLE_SEP_RE.match(row_line):
                    continue
                cells = _split_table_cells(row_line)
                if len(cells) > 0:
                    data_rows.append(cells)

            table_rows = len(data_rows)
            table_cols = max([len(c) for c in data_rows], default=0)

            for row_line in block_lines:
                row_line = row_line.strip()
                if TABLE_SEP_RE.match(row_line):
                    append_table_record(row_line)
                    continue

                cells = _split_table_cells(row_line)
                if len(cells) == 0:
                    append_table_record(row_line)
                    continue

                rule_name, tokens = _classify_table_row_rule(
                    cells,
                    table_rows,
                    table_cols,
                    threshold=TABLE_TOKEN_THRESHOLD,
                )

                if rule_name == 'rule_b_all_non_empty_oversized':
                    handle_oversized_cells(cells, tokens, depth=depth, max_depth=max_depth)
                    continue

                if rule_name == 'rule_a_single_row_or_col_with_oversized':
                    handle_oversized_cells(cells, tokens, depth=depth, max_depth=max_depth)
                    continue

                append_table_record(' | '.join(cells))

        lines = self.mkdown.splitlines()
        i = 0
        code_block_buffer = []
        while i < len(lines):
            line = lines[i].rstrip('\n')
            stripped = line.strip()

            if stripped.startswith('```'):
                if in_code_block:
                    process_code_block_lines(code_block_buffer, depth=0, max_depth=6)
                    code_block_buffer.clear()
                    in_code_block = False
                else:
                    flush_paragraph()
                    in_code_block = True
                i += 1
                continue
            if in_code_block:
                code_text = stripped
                if code_text:
                    code_block_buffer.append(code_text)
                i += 1
                continue

            if len(stripped) == 0:
                flush_paragraph()
                i += 1
                continue

            hmatch = HEADER_RE.match(line)
            if hmatch:
                flush_paragraph()
                header_level = len(hmatch.group(1))
                header_text = hmatch.group(2).strip()
                if header_text:
                    heading_stack[header_level] = header_text
                    for level in range(header_level + 1, 7):
                        heading_stack.pop(level, None)
                i += 1
                continue

            if is_table_candidate(line):
                flush_paragraph()
                block_lines = []
                while i < len(lines):
                    cur = lines[i].rstrip('\n')
                    cur_stripped = cur.strip()
                    if len(cur_stripped) == 0:
                        break
                    if cur_stripped.startswith('```'):
                        break
                    if HEADER_RE.match(cur):
                        break
                    if LIST_RE.match(cur) and not is_table_candidate(cur):
                        break
                    if is_table_candidate(cur):
                        block_lines.append(cur_stripped)
                        i += 1
                        continue
                    break

                process_table_block(block_lines, depth=0, max_depth=6)
                continue

            lmatch = LIST_RE.match(line)
            if lmatch:
                flush_paragraph()
                item_text = lmatch.group(1).strip()
                if item_text:
                    append_list_record(item_text)
                i += 1
                continue

            paragraph_buffer.append(stripped)
            i += 1

        flush_paragraph()
        if in_code_block and len(code_block_buffer) > 0:
            process_code_block_lines(code_block_buffer, depth=0, max_depth=6)
        output = _apply_synthetic_list_merge(output)
        return TextPostProcessor(output).post_process()


def _sentence_prefix_from_filename(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    prefix = re.sub(r'[^A-Za-z0-9]+', '_', base).strip('_').lower()
    return prefix if prefix else 'doc'


def _txt_to_records_by_newline(raw_text):
    records = []
    normalized = str(raw_text or '').replace('\r\n', '\n').replace('\r', '\n')
    for line in normalized.split('\n'):
        if len(str(line).strip()) == 0:
            continue
        records.append(
            {
                'section': 'General',
                'sentence': line,
                'source_type': 'paragraph',
            }
        )
    return records


def process_file(filename, output_dir=None, force=False):
    import time
    _t0 = time.monotonic()
    ext = os.path.splitext(filename)[1].lower()
    if ext not in (MD_SUFFIXES | TXT_SUFFIXES | HTML_SUFFIXES):
        return None

    raw_text_for_lang = ''
    raw_html = None
    html_records = None
    if ext in MD_SUFFIXES:
        raw_text_for_lang = _load_markdown_text(filename)
    elif ext in TXT_SUFFIXES:
        raw_text_for_lang = _load_markdown_text(filename)
    elif ext in HTML_SUFFIXES:
        raw_html = _load_markdown_text(filename)
        raw_text_for_lang = BeautifulSoup(raw_html, HTML_PARSER).get_text(' ', strip=True)

    if len(str(raw_text_for_lang).strip()) == 0:
        return {
            'file': os.path.abspath(filename),
            'lang': 'unknown',
            'is_en': False,
            'kept': False,
            'char_count': 0,
            'empty_input': True,
        }

    lang_code = _detect_language_code(raw_text_for_lang)
    detection = {
        'file': os.path.abspath(filename),
        'lang': lang_code,
        'is_en': bool(lang_code == 'en'),
        'kept': False,
        'char_count': len(raw_text_for_lang),
    }
    if lang_code != 'en':
        print(f"skip_non_en: {os.path.basename(filename)} lang={lang_code}")
        return detection

    output_filename = '{}.jsonl'.format(os.path.splitext(os.path.basename(filename))[0])
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, output_filename)

    if (not force) and os.path.isfile(output_filename) and os.path.getsize(output_filename) > 0:
        detection['kept'] = True
        detection['skipped_existing'] = True
        return detection

    if ext in TXT_SUFFIXES:
        res = _txt_to_records_by_newline(raw_text_for_lang)
    elif ext in HTML_SUFFIXES:
        preprocessed_html = preprocess_html_before_parse(raw_html or '')
        html_records = _parse_html_records(preprocessed_html)
        res = html_records if html_records is not None else []
    elif ext in MD_SUFFIXES:
        processor = Preprocessor(filename=filename)
        res = processor.parse()
    else:
        processor = Preprocessor(mkdown_text=raw_md)
        res = processor.parse()
    prefix = _sentence_prefix_from_filename(filename)

    detection['sentences'] = len(res)
    if len(res) == 0:
        if os.path.isfile(output_filename):
            try:
                os.remove(output_filename)
            except OSError:
                pass
        detection['kept'] = False
        detection['skipped_existing'] = False
        detection['empty_parsed'] = True
        detection['elapsed_sec'] = round(time.monotonic() - _t0, 1)
        print(f"  processed {os.path.basename(filename)}: 0 sentences, {detection['elapsed_sec']}s (no jsonl written)")
        return detection

    with codecs.open(output_filename, 'w', 'utf-8') as outputfile:
        for idx, rec in enumerate(res, start=1):
            record = {
                'sentence_id': f'{prefix}_{idx:04d}',
                'section': rec.get('section', 'General'),
                'sentence': _normalize_unicode_text(rec.get('sentence', '')),
                'source_type': rec.get('source_type', 'paragraph'),
            }
            outputfile.write(json.dumps(record, ensure_ascii=False))
            outputfile.write('\n')
    detection['kept'] = True
    detection['skipped_existing'] = False
    detection['elapsed_sec'] = round(time.monotonic() - _t0, 1)
    print(f"  processed {os.path.basename(filename)}: {len(res)} sentences, {detection['elapsed_sec']}s")
    return detection


def process_directory(directory, output_dir=None, force=False, workers=6):
    all_files = []
    for root, _dirs, files in os.walk(directory):
        for f in files:
            if os.path.splitext(f)[1].lower() in (MD_SUFFIXES | TXT_SUFFIXES | HTML_SUFFIXES):
                all_files.append(os.path.join(root, f))

    all_files = sorted(all_files)
    detections = []
    if len(all_files) == 0:
        return detections

    max_workers = max(1, int(workers))
    pbar = tqdm(total=len(all_files), desc='file2jsonl', unit='file') if tqdm is not None else None
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_file, filepath, output_dir, force): filepath for filepath in all_files
        }
        for fut in concurrent.futures.as_completed(future_to_file):
            filepath = future_to_file[fut]
            if pbar is None:
                print(filepath)
            try:
                rec = fut.result()
            except Exception as exc:
                print(f'failed: {filepath} error={exc}')
                rec = None
            if rec is not None:
                detections.append(rec)
            if pbar is not None:
                pbar.update(1)
    if pbar is not None:
        pbar.close()
    return detections


def write_language_outputs(detections, lang_log_path=None, lang_dist_path=None):
    if lang_log_path:
        parent = os.path.dirname(lang_log_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with codecs.open(lang_log_path, 'w', 'utf-8') as fp:
            for rec in detections:
                fp.write(json.dumps(rec, ensure_ascii=False))
                fp.write('\n')

    if lang_dist_path:
        parent = os.path.dirname(lang_dist_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        total = len(detections)
        counts = {}
        for rec in detections:
            code = rec.get('lang', 'unknown')
            counts[code] = counts.get(code, 0) + 1
        rows = []
        for code, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            ratio = (count / total) if total > 0 else 0.0
            rows.append(
                {
                    'lang': code,
                    'count': count,
                    'ratio': ratio,
                }
            )
        payload = {
            'total_files': total,
            'kept_en_files': sum([1 for r in detections if r.get('kept')]),
            'distribution': rows,
        }
        with codecs.open(lang_dist_path, 'w', 'utf-8') as fp:
            fp.write(json.dumps(payload, ensure_ascii=False, indent=2))
            fp.write('\n')
