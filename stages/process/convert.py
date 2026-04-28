# -*- coding: utf-8 -*-
"""Convert document files (.pdf/.docx) to markdown using markitdown."""

from __future__ import annotations

import argparse
import concurrent.futures
import tempfile
from pathlib import Path
from typing import Iterable, List, Tuple

try:
    from markitdown import MarkItDown
except ImportError:  # pragma: no cover
    MarkItDown = None


BASE_DIR = Path(__file__).resolve().parent
DOC_SUFFIXES = {".pdf", ".docx"}
SUPPORTED_SUFFIXES = set(DOC_SUFFIXES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert document files (.pdf/.docx) to markdown."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Input file or directory. Supported suffixes: .pdf/.docx",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=BASE_DIR / "output" / "md",
        help=f"Output markdown directory. Default: {BASE_DIR / 'output' / 'md'}",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Number of worker threads for conversion. Default: 6",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="If input is a directory, scan only current directory (no recursion).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .md outputs.",
    )
    return parser.parse_args()


def collect_input_files(input_path: Path, recursive: bool) -> List[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in SUPPORTED_SUFFIXES else []
    if not input_path.is_dir():
        return []

    iterator: Iterable[Path] = input_path.rglob("*") if recursive else input_path.glob("*")
    files = [
        p
        for p in iterator
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return sorted(files)


def output_md_path(src: Path, input_root: Path, output_root: Path) -> Path:
    if input_root.is_file():
        return output_root / f"{src.stem}.md"
    rel = src.relative_to(input_root)
    return (output_root / rel).with_suffix(".md")


def _get_converter() -> MarkItDown:
    if MarkItDown is None:
        raise RuntimeError("markitdown is not installed. Please install dependencies first.")
    return MarkItDown()


def convert_markitdown_file(src: Path) -> str:
    converter = _get_converter()
    result = converter.convert(str(src))
    text = getattr(result, "text_content", None)
    return text or ""


def convert_html_table_fragment(table_html: str) -> str:
    """Convert a raw <table>...</table> HTML fragment into markdown table text."""
    fragment = f"<html><body>{table_html}</body></html>"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", encoding="utf-8", delete=True) as tmp:
        tmp.write(fragment)
        tmp.flush()
        return convert_markitdown_file(Path(tmp.name)).strip()


def convert_one(src: Path, dst: Path) -> Tuple[str, str]:
    suffix = src.suffix.lower()
    if suffix not in DOC_SUFFIXES:
        return ("skipped", f"unsupported suffix: {src}")

    md_text = convert_markitdown_file(src)
    if not md_text:
        return ("empty", f"empty output: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(md_text, encoding="utf-8")
    return ("success", f"written: {dst}")


def run_convert(
    input_path: Path,
    output_dir: Path,
    workers: int = 6,
    recursive: bool = True,
    overwrite: bool = False,
) -> dict:
    """Run conversion on input path, return stats dict."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    files = collect_input_files(input_path, recursive=recursive)
    if not files:
        return {"success": 0, "empty": 0, "failed": 0, "skipped_existing": 0, "skipped_unsupported": 0}

    output_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = 0
    empty = 0
    skipped_existing = 0
    skipped_unsupported = 0
    empty_files: List[str] = []

    jobs = []
    for src in files:
        dst = output_md_path(src=src, input_root=input_path, output_root=output_dir)
        if dst.exists() and not overwrite:
            skipped_existing += 1
            continue
        jobs.append((src, dst))

    if not jobs:
        return {"success": success, "empty": empty, "failed": failed, "skipped_existing": skipped_existing, "skipped_unsupported": skipped_unsupported}

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(convert_one, src, dst): (src, dst) for (src, dst) in jobs}
        for fut in concurrent.futures.as_completed(future_map):
            src, _dst = future_map[fut]
            try:
                status, _message = fut.result()
            except Exception as exc:
                failed += 1
                continue

            if status == "success":
                success += 1
            elif status == "empty":
                empty += 1
                empty_files.append(str(src))
            elif status == "skipped":
                skipped_unsupported += 1
            else:
                failed += 1

    return {"success": success, "empty": empty, "failed": failed, "skipped_existing": skipped_existing, "skipped_unsupported": skipped_unsupported}
