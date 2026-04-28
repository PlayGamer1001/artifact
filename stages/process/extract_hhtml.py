# -*- coding: utf-8 -*-
"""Extract article content from HTML using readability (lxml-based).

This is step 1 of the process pipeline.
PDF/DOCX/TXT are copied directly without modification.
"""

from __future__ import annotations

import argparse
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path


HTML_EXTENSIONS = {".html", ".htm"}
COPY_EXTENSIONS = {".pdf", ".docx", ".txt"}
WHITE_SPACE_STYLE_RE = re.compile(r"white-space\s*:\s*(pre|pre-line|pre-wrap)", re.IGNORECASE)
BODY_WRAPPER_RE = re.compile(r"(?is)<body\b[^>]*>(.*)</body>")


def _normalize_text_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _extract_whitelist_style(style_value: str) -> str | None:
    value = str(style_value or "")
    m = WHITE_SPACE_STYLE_RE.search(value)
    if m is None:
        return None
    return f"white-space:{m.group(1).lower()}"


def _build_source_style_index(raw_html: str) -> dict:
    from lxml import html as lxml_html
    index: dict = {}
    try:
        root = lxml_html.fromstring(raw_html)
    except Exception:
        return index
    for node in root.iter():
        style = node.attrib.get("style")
        if not style:
            continue
        keep_style = _extract_whitelist_style(style)
        if keep_style is None:
            continue
        text_key = _normalize_text_for_match(node.text_content())
        if not text_key:
            continue
        if text_key not in index:
            index[text_key] = keep_style
    return index


def _apply_missing_whitelist_style(content_html: str, source_style_index: dict) -> tuple[str, int]:
    from lxml import html as lxml_html
    if not source_style_index:
        return content_html, 0
    wrapped = f"<div>{content_html}</div>"
    try:
        root = lxml_html.fromstring(wrapped)
    except Exception:
        return content_html, 0
    added = 0
    for node in root.iter():
        tag = getattr(node, "tag", None)
        if not isinstance(tag, str):
            continue
        text_key = _normalize_text_for_match(node.text_content())
        if not text_key:
            continue
        target_style = source_style_index.get(text_key)
        if target_style is None:
            continue
        cur_style = str(node.attrib.get("style", ""))
        cur_keep = _extract_whitelist_style(cur_style)
        if cur_keep is not None:
            continue
        merged_style = f"{cur_style.strip().rstrip(';')}; {target_style}".strip()
        merged_style = merged_style.lstrip("; ").strip()
        node.attrib["style"] = merged_style
        added += 1
    inner_html = "".join(
        lxml_html.tostring(child, encoding="unicode", method="html")
        for child in root
    )
    return inner_html, added


def build_output_html(title: str, content_html: str) -> str:
    content = str(content_html or "").strip()
    m = BODY_WRAPPER_RE.search(content)
    if m is not None:
        content = m.group(1).strip()
    return f"""<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
</head>
<body>
{content}
</body>
</html>"""


def get_document_class():
    global _DOCUMENT_CLASS
    if _DOCUMENT_CLASS is None:
        from readability import Document
        _DOCUMENT_CLASS = Document
    return _DOCUMENT_CLASS


_DOCUMENT_CLASS = None


def process_file(input_file: Path, output_file: Path) -> int:
    Document = get_document_class()
    with input_file.open("r", encoding="utf-8", errors="ignore") as f:
        html = f.read()
    source_style_index = _build_source_style_index(html)
    doc = Document(html)
    title = doc.title() or input_file.stem
    content_html = doc.summary()
    content_html, added_style_count = _apply_missing_whitelist_style(content_html, source_style_index)
    final_html = build_output_html(title, content_html)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        f.write(final_html)
    return added_style_count


def should_process_html(input_file: Path) -> bool:
    return input_file.suffix.lower() in HTML_EXTENSIONS


def should_copy_directly(input_file: Path) -> bool:
    return input_file.suffix.lower() in COPY_EXTENSIONS


def process_one(
    input_dir: Path,
    output_dir: Path,
    input_file: Path,
    recursive: bool,
) -> tuple[bool, str, int]:
    if input_file.suffix.lower() in HTML_EXTENSIONS:
        output_file = output_dir / input_file.relative_to(input_dir) if recursive else output_dir / input_file.name
        try:
            added_style_count = process_file(input_file, output_file)
            return True, f"[HTML][style_added={added_style_count}]", added_style_count
        except Exception as exc:
            return False, f"[FAIL] {input_file}: {exc}", 0
    elif input_file.suffix.lower() in COPY_EXTENSIONS:
        output_file = output_dir / input_file.relative_to(input_dir) if recursive else output_dir / input_file.name
        output_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_file, output_file)
        return True, "[COPY]", 0
    return True, "[SKIP]", 0


def run_extract(
    input_dir: Path,
    output_dir: Path,
    recursive: bool = True,
    workers: int = 28,
    executor: str = "process",
) -> dict:
    """Run readability extraction on input directory.

    Returns stats dict.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if recursive:
        files = [p for p in input_dir.rglob("*") if p.is_file()]
    else:
        files = [p for p in input_dir.iterdir() if p.is_file()]
    files = sorted(set(files))

    if not files:
        return {"success": 0, "skipped": 0, "failed": 0, "total_style_added": 0}

    success = 0
    skipped = 0
    failed = 0
    total_style_added = 0

    file_to_name = {f: f.name for f in files}
    if executor == "serial":
        for input_file in files:
            ok, message, style_added = process_one(input_dir, output_dir, input_file, recursive)
            print(f"  {message} {file_to_name[input_file]}")
            if message.startswith("[SKIP]"):
                skipped += 1
            elif ok:
                success += 1
                total_style_added += style_added
            else:
                failed += 1
    else:
        executor_cls = ProcessPoolExecutor if executor == "process" else ThreadPoolExecutor
        with executor_cls(max_workers=workers) as ex:
            futures = {
                ex.submit(process_one, input_dir, output_dir, input_file, recursive): input_file
                for input_file in files
            }
            for future in as_completed(futures):
                input_file = futures[future]
                ok, message, style_added = future.result()
                print(f"  {message} {file_to_name[input_file]}")
                if message.startswith("[SKIP]"):
                    skipped += 1
                elif ok:
                    success += 1
                    total_style_added += style_added
                else:
                    failed += 1

    return {
        "success": success,
        "skipped": skipped,
        "failed": failed,
        "total_style_added": total_style_added,
    }
