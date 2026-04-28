# -*- coding: utf-8 -*-
"""Clean HTML payloads: remove data URI from src/href/style attributes (inline cleanup only).

This is step 0 of the process pipeline.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


URL_DATA_PATTERN = re.compile(r"url\(\s*([\"']?)data:[^)]*\1\s*\)", re.IGNORECASE)
RESOURCE_ATTR_QUOTED_PATTERN = re.compile(
    r"(\b(?:src|href|poster|data-src)\s*=\s*)([\"'])(?:(?!\2)[\s\S])*?data:(?:(?!\2)[\s\S])*?\2",
    re.IGNORECASE | re.DOTALL,
)
RESOURCE_ATTR_UNQUOTED_PATTERN = re.compile(
    r"(\b(?:src|href|poster|data-src)\s*=\s*)data:[^\s>]+",
    re.IGNORECASE,
)
SRCSET_ATTR_QUOTED_PATTERN = re.compile(r"(\bsrcset\s*=\s*)([\"'])(.*?)\2", re.IGNORECASE | re.DOTALL)
SRCSET_ATTR_UNQUOTED_PATTERN = re.compile(r"(\bsrcset\s*=\s*)(?![\"'])([^\s>]+)", re.IGNORECASE)
STYLE_ATTR_QUOTED_PATTERN = re.compile(r"(\bstyle\s*=\s*)([\"'])(.*?)\2", re.IGNORECASE | re.DOTALL)
STYLE_ATTR_UNQUOTED_PATTERN = re.compile(r"(\bstyle\s*=\s*)(?![\"'])([^\s>]+)", re.IGNORECASE)

ICON_LINK_PATTERN = re.compile(
    r"<link\b[^>]*\brel\s*=\s*(?:icon|\"[^\"]*\bicon\b[^\"]*\"|'[^']*\bicon\b[^']*')[^>]*>",
    re.IGNORECASE,
)
NOSCRIPT_BLOCK_PATTERN = re.compile(r"<noscript\b[^>]*>.*?</noscript>", re.IGNORECASE | re.DOTALL)
SCRIPT_BLOCK_PATTERN = re.compile(r"<script\b[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL)
STYLE_TAG_BLOCK_PATTERN = re.compile(r"<style\b[^>]*>.*?</style>", re.IGNORECASE | re.DOTALL)


def _clean_srcset_value(srcset_value: str):
    candidates = [item.strip() for item in srcset_value.split(",") if item.strip()]
    kept = []
    removed = 0
    for item in candidates:
        parts = item.split()
        if not parts:
            continue
        url = parts[0]
        if url.lower().startswith("data:"):
            removed += 1
            continue
        kept.append(item)
    return ", ".join(kept), removed


def _remove_common_noise(html_content: str):
    stats = {
        "icon_links": 0,
        "noscript_blocks": 0,
        "script_blocks": 0,
        "style_blocks": 0,
    }
    html_content, n = ICON_LINK_PATTERN.subn("", html_content)
    stats["icon_links"] += n
    html_content, n = NOSCRIPT_BLOCK_PATTERN.subn("", html_content)
    stats["noscript_blocks"] = n
    html_content, n = SCRIPT_BLOCK_PATTERN.subn("", html_content)
    stats["script_blocks"] = n
    html_content, n = STYLE_TAG_BLOCK_PATTERN.subn("", html_content)
    stats["style_blocks"] = n
    return html_content, stats


def clean_html_content(html_content: str):
    stats = {
        "resource_attrs": 0,
        "srcset_items": 0,
        "style_attr_urls": 0,
        "icon_links": 0,
        "noscript_blocks": 0,
        "script_blocks": 0,
        "style_blocks": 0,
    }

    def replace_resource_attr(match):
        stats["resource_attrs"] += 1
        prefix, quote = match.groups()
        return f"{prefix}{quote}{quote}"

    cleaned_html = RESOURCE_ATTR_QUOTED_PATTERN.sub(replace_resource_attr, html_content)

    def replace_resource_attr_unquoted(match):
        stats["resource_attrs"] += 1
        prefix = match.group(1)
        return f'{prefix}""'

    cleaned_html = RESOURCE_ATTR_UNQUOTED_PATTERN.sub(replace_resource_attr_unquoted, cleaned_html)

    def replace_srcset_attr(match):
        prefix, quote, value = match.groups()
        cleaned_value, removed = _clean_srcset_value(value)
        stats["srcset_items"] += removed
        return f"{prefix}{quote}{cleaned_value}{quote}"

    cleaned_html = SRCSET_ATTR_QUOTED_PATTERN.sub(replace_srcset_attr, cleaned_html)

    def replace_srcset_attr_unquoted(match):
        prefix, value = match.groups()
        cleaned_value, removed = _clean_srcset_value(value)
        stats["srcset_items"] += removed
        return f'{prefix}"{cleaned_value}"'

    cleaned_html = SRCSET_ATTR_UNQUOTED_PATTERN.sub(replace_srcset_attr_unquoted, cleaned_html)

    def _replace_style_value(prefix: str, quote: str, value: str):
        new_value, removed = URL_DATA_PATTERN.subn('url("")', value)
        stats["style_attr_urls"] += removed
        return f"{prefix}{quote}{new_value}{quote}"

    def replace_style_attr(match):
        prefix, quote, value = match.groups()
        return _replace_style_value(prefix, quote, value)

    cleaned_html = STYLE_ATTR_QUOTED_PATTERN.sub(replace_style_attr, cleaned_html)

    def replace_style_attr_unquoted(match):
        prefix, value = match.groups()
        return _replace_style_value(prefix, '"', value)

    cleaned_html = STYLE_ATTR_UNQUOTED_PATTERN.sub(replace_style_attr_unquoted, cleaned_html)
    cleaned_html, noise_stats = _remove_common_noise(cleaned_html)
    stats.update(noise_stats)
    return cleaned_html, stats


def clean_html_file(input_html_path: str, output_html_path: str):
    with open(input_html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    cleaned_html, stats = clean_html_content(html_content)
    os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(cleaned_html)
    original_size = len(html_content) / (1024 * 1024)
    cleaned_size = len(cleaned_html) / (1024 * 1024)
    ratio = (1 - cleaned_size / original_size) * 100 if original_size > 0 else 0.0
    return {
        "input": input_html_path,
        "output": output_html_path,
        "original_mb": original_size,
        "cleaned_mb": cleaned_size,
        "ratio": ratio,
        "stats": stats,
    }


def _process_html_task(task):
    input_file, output_file = task
    try:
        result = clean_html_file(str(input_file), str(output_file))
        return {"ok": True, "result": result}
    except Exception as exc:
        return {"ok": False, "input": str(input_file), "output": str(output_file), "error": str(exc)}


def run_clean(
    input_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
    workers: int = 1,
) -> dict:
    """Run HTML cleaning on input directory.

    Returns stats dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    failed = 0
    copied_docs_txt = 0
    skipped_other = 0
    skipped_under_files_dir = 0
    skipped_existing = 0
    html_tasks = []

    for file_path in sorted(input_dir.rglob("*")):
        if not file_path.is_file():
            continue
        relative = file_path.relative_to(input_dir)
        if any(part.lower().endswith("_files") for part in relative.parts[:-1]):
            skipped_under_files_dir += 1
            continue
        suffix = file_path.suffix.lower()
        target = output_dir / relative
        if target.exists() and not overwrite:
            skipped_existing += 1
            continue
        if suffix in {".html", ".htm"}:
            html_tasks.append((file_path, target))
        elif suffix in {".pdf", ".docx", ".txt"}:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, target)
            copied_docs_txt += 1
        else:
            skipped_other += 1

    if workers <= 1:
        for input_file, target in html_tasks:
            outcome = _process_html_task((input_file, target))
            if outcome["ok"]:
                result = outcome["result"]
                processed += 1
                print(
                    f"  [OK] {Path(result['input']).relative_to(input_dir)} -> "
                    f"{result['original_mb']:.2f}MB -> {result['cleaned_mb']:.2f}MB ({result['ratio']:.1f}%)"
                )
            else:
                failed += 1
                print(f"  [ERR] {Path(outcome['input']).relative_to(input_dir)}: {outcome['error']}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_process_html_task, task) for task in html_tasks]
            for future in as_completed(futures):
                outcome = future.result()
                if outcome["ok"]:
                    result = outcome["result"]
                    processed += 1
                    print(
                        f"  [OK] {Path(result['input']).relative_to(input_dir)} -> "
                        f"{result['original_mb']:.2f}MB -> {result['cleaned_mb']:.2f}MB ({result['ratio']:.1f}%)"
                    )
                else:
                    failed += 1
                    print(f"  [ERR] {Path(outcome['input']).relative_to(input_dir)}: {outcome['error']}")

    return {
        "cleaned_html": processed,
        "clean_failed": failed,
        "copied_pdf_docx_txt": copied_docs_txt,
        "skipped_existing": skipped_existing,
        "skipped_under_files_dir": skipped_under_files_dir,
        "skipped_other": skipped_other,
    }
