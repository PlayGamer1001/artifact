# -*- coding: utf-8 -*-
"""Process stage: convert raw documents to JSONL sentences via markitdown + parsing.

Pipeline (mirrors reference run_pipeline.sh):
    0. Clean HTML data URI payloads (clean_hhtml.py)
    1. Extract article content from HTML (extract_hhtml.py)
    2. Convert DOCX/PDF -> MD (convert.py)
    3. Parse HTML/MD/TXT -> JSONL (parse.py) — with language detection
    4. Token distribution stats (distrib.py)
    [MD intermediate files are deleted after JSONL conversion]

Input:  raw HTML/DOCX/PDF/TXT files
Output: JSONL sentence files + language detection + token distribution stats
"""

from __future__ import annotations

import json
import traceback
import shutil
from pathlib import Path
from typing import Any, Dict

from .. import config
from ..lib.runtime_logger import RunLogger, append_run_error_log

from .clean_hhtml import run_clean
from .extract_hhtml import run_extract
from .convert import run_convert
from .parse import process_directory as parse_directory
from .distrib import run_token_distribution


SUPPORTED_SUFFIXES = {".html", ".htm", ".md", ".txt", ".pdf", ".docx"}
DOC_SUFFIXES = {".pdf", ".docx"}


def _collect_input_files(input_dir: Path, recursive: bool) -> list[Path]:
    """Collect all supported input files from input directory."""
    if not input_dir.exists():
        return []
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)


def _separate_doc_and_direct(files: list[Path]) -> tuple[list[Path], list[Path]]:
    doc, direct = [], []
    for f in files:
        if f.suffix.lower() in DOC_SUFFIXES:
            doc.append(f)
        else:
            direct.append(f)
    return doc, direct


def _write_lang_log(detections: list[dict], path: Path) -> None:
    """Write language detection results to JSONL (one JSON object per line)."""
    if not detections:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        for rec in detections:
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_empty_file_log(detections: list[dict], path: Path) -> None:
    """Write source filenames whose parsing yielded 0 sentences."""
    names = []
    for rec in detections:
        if int(rec.get("sentences", 0) or 0) != 0:
            continue
        src = str(rec.get("file", "")).strip()
        if not src:
            continue
        names.append(Path(src).name)
    uniq = sorted(set(names))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        for name in uniq:
            fp.write(name + "\n")


def run_process_stage(
    output_dir: Path,
    target_dir_name: str,
    base_url: str,
    model: str,
    mode: str,
    concurrency: int,
    files_per_batch: int,
    logger: RunLogger,
    run_error_log: Path,
    input_dir: Path | None = None,
) -> Dict[str, int]:
    """Run the full process stage: convert documents to JSONL sentences."""
    if input_dir is None:
        input_dir = config.PROCESS_INPUT_DIR

    process_root = output_dir / target_dir_name / "process"
    md_dir = process_root / "md"
    # JSONL goes to data/jsonl/ so annotate/extract/normalize can read without extra flags
    jsonl_dir = input_dir.parent / "jsonl"
    eval_dir = process_root / "evaluation"

    process_root.mkdir(parents=True, exist_ok=True)

    logger.paths.update({
        "input_dir": str(input_dir),
        "process_root": str(process_root),
        "md_dir": str(md_dir),
        "jsonl_dir": str(jsonl_dir),
        "eval_dir": str(eval_dir),
    })

    all_files = _collect_input_files(input_dir, recursive=True)
    if not all_files:
        suffixes = ", ".join(sorted(SUPPORTED_SUFFIXES))
        if not input_dir.exists():
            msg = (
                f"[ERROR] process input directory not found: {input_dir}\n"
                f"        Please create the directory or pass --process-input-dir.\n"
                f"        Supported file suffixes: {suffixes}"
            )
        else:
            msg = (
                f"[ERROR] no supported input files found under: {input_dir}\n"
                f"        Supported file suffixes: {suffixes}"
            )
        print(msg)
        append_run_error_log(
            run_error_log,
            {
                "stage": "process",
                "step": "collect_input_files",
                "error": msg,
            },
        )
        logger.event("run_done", {"reason": "no_input_files"})
        return {"files_total": 0, "files_kept": 0, "records_total": 0}

    doc_files, direct_files = _separate_doc_and_direct(all_files)
    logger.event("batch_start", {
        "files_total": len(all_files),
        "doc_files": len(doc_files),
        "direct_files": len(direct_files),
    })

    eval_dir.mkdir(parents=True, exist_ok=True)
    empty_file_log_path = eval_dir / "empty_sentence_files.txt"
    tmp_lang_html = eval_dir / "language_detection_html.tmp.jsonl"
    tmp_lang_md = eval_dir / "language_detection_md.tmp.jsonl"

    # -------------------------------------------------------------------------
    # Step 0: Clean HTML data URI payloads (clean_hhtml.py)
    # -------------------------------------------------------------------------
    print(f"[0/5] clean html data uri: {input_dir}")
    clean_stats = run_clean(
        input_dir=input_dir,
        output_dir=process_root / "cleaned",
        overwrite=True,
        workers=concurrency,
    )
    print(f"      clean done: {clean_stats}")
    logger.event("step_clean_done", clean_stats)

    # Use cleaned HTML for subsequent steps
    cleaned_input_dir = process_root / "cleaned"

    # -------------------------------------------------------------------------
    # Step 1: Extract article content from HTML (extract_hhtml.py)
    # -------------------------------------------------------------------------
    print(f"[1/5] extract article content (html): {cleaned_input_dir}")
    extract_stats = run_extract(
        input_dir=cleaned_input_dir,
        output_dir=process_root / "extracted",
        recursive=True,
        workers=concurrency,
        executor="process",
    )
    print(f"      extract done: {extract_stats}")
    logger.event("step_extract_done", extract_stats)

    # Use extracted HTML for parsing; PDF/DOCX go through file2md
    extracted_html_dir = process_root / "extracted"

    # -------------------------------------------------------------------------
    # Step 2: Convert DOCX/PDF -> MD (6-threaded markitdown)
    # -------------------------------------------------------------------------
    print(f"[2/5] file2md (pdf/docx only): {input_dir} -> {md_dir}")
    md_files: list[Path] = []
    if doc_files:
        logger.event("step_convert_start", {"doc_files": len(doc_files)})
        convert_stats = run_convert(
            input_path=input_dir,
            output_dir=md_dir,
            workers=concurrency,
            recursive=True,
            overwrite=False,
        )
        print(f"      file2md done: {convert_stats}")
        logger.event("step_convert_done", convert_stats)
        md_files = [p for p in md_dir.rglob("*.md") if p.is_file()]
    else:
        print("      (no docx/pdf files found, skipped)")

    # -------------------------------------------------------------------------
    # Step 3a: Parse direct files (HTML/MD/TXT) -> JSONL
    # -------------------------------------------------------------------------
    print(f"[3/5] preprocess html/txt: {extracted_html_dir} -> {jsonl_dir}")
    detections_html: list[dict] = []
    html_to_parse = [f for f in _collect_input_files(extracted_html_dir, recursive=True) if f.suffix.lower() in {".html", ".htm"}]
    if html_to_parse:
        logger.event("step_parse_direct_start", {"direct_files": len(html_to_parse)})
        try:
            detections_html = parse_directory(
                str(extracted_html_dir),
                output_dir=str(jsonl_dir),
                force=True,
                workers=concurrency,
            )
            kept_html = sum(1 for r in detections_html if r.get("kept"))
            print(f"      {len(detections_html)} files processed, {kept_html} kept (en)")
            _write_lang_log(detections_html, tmp_lang_html)
        except Exception as err:
            print(f"      FAILED: {err}")
            logger.error(
                error_type="parse_direct_failed",
                message=str(err),
                traceback_text=traceback.format_exc(),
            )
            append_run_error_log(run_error_log, {
                "stage": "process",
                "step": "parse_direct",
                "error": str(err),
            })
            detections_html = []
        logger.event("step_parse_direct_done", {"files": len(detections_html)})
    else:
        print("      (no html/htm files found in extracted, skipped)")

    # -------------------------------------------------------------------------
    # Step 3b: Parse converted MD files -> JSONL, then delete MD dir
    # -------------------------------------------------------------------------
    detections_md: list[dict] = []
    if md_files:
        print(f"[4/5] preprocess md (from pdf/docx): {md_dir} -> {jsonl_dir}")
        logger.event("step_parse_md_start", {"md_files": len(md_files)})
        try:
            detections_md = parse_directory(
                str(md_dir),
                output_dir=str(jsonl_dir),
                force=True,
                workers=concurrency,
            )
            kept_md = sum(1 for r in detections_md if r.get("kept"))
            print(f"      {len(detections_md)} files processed, {kept_md} kept (en)")
            _write_lang_log(detections_md, tmp_lang_md)
        except Exception as err:
            print(f"      FAILED: {err}")
            logger.error(
                error_type="parse_md_failed",
                message=str(err),
                traceback_text=traceback.format_exc(),
            )
            append_run_error_log(run_error_log, {
                "stage": "process",
                "step": "parse_md",
                "error": str(err),
            })
            detections_md = []
        logger.event("step_parse_md_done", {"files": len(detections_md)})

        # Delete MD directory after JSONL conversion is done
        try:
            shutil.rmtree(md_dir)
            print(f"      md files deleted: {md_dir}")
            logger.event("md_dir_deleted", {"path": str(md_dir)})
        except Exception as err:
            print(f"      WARNING: failed to delete md dir: {err}")
            logger.error(
                error_type="md_dir_delete_failed",
                message=str(err),
                traceback_text=traceback.format_exc(),
            )
            append_run_error_log(run_error_log, {
                "stage": "process",
                "step": "delete_md_dir",
                "error": str(err),
            })
    else:
        print(f"[4/5] preprocess md: (no md files to process, skipped)")

    # -------------------------------------------------------------------------
    # Step 4: Token distribution analysis
    # -------------------------------------------------------------------------
    print(f"[5/5] token distribution: {jsonl_dir} -> {eval_dir}")
    jsonl_files = list(jsonl_dir.rglob("*.jsonl")) if jsonl_dir.exists() else []
    if jsonl_files:
        logger.event("step_distrib_start", {"jsonl_files": len(jsonl_files)})
        distrib_path = eval_dir / "token_distribution_summary.json"
        gt512_path = eval_dir / "token_gt512_records.jsonl"
        try:
            summary = run_token_distribution(
                input_path=jsonl_dir,
                output_path=distrib_path,
                gt512_output=gt512_path,
            )
            print(
                f"      {summary.get('files_scanned', 0)} files, "
                f"{summary.get('valid_sentences', 0)} sentences, "
                f"p50={summary.get('basic_stats', {}).get('p50', 0):.0f} tokens"
            )
            logger.event("step_distrib_done", {
                "files_scanned": summary.get("files_scanned", 0),
                "records_scanned": summary.get("records_scanned", 0),
                "valid_sentences": summary.get("valid_sentences", 0),
            })
        except Exception as err:
            print(f"      FAILED: {err}")
            logger.error(
                error_type="token_distribution_failed",
                message=str(err),
                traceback_text=traceback.format_exc(),
            )
            append_run_error_log(run_error_log, {
                "stage": "process",
                "step": "token_distribution",
                "error": str(err),
            })
            logger.event("step_distrib_done", {"error": str(err)})

    # -------------------------------------------------------------------------
    # Final counts
    # -------------------------------------------------------------------------
    all_detections = detections_html + detections_md
    _write_empty_file_log(all_detections, empty_file_log_path)
    files_kept = sum(1 for r in all_detections if r.get("kept"))
    records_total = sum(r.get("sentences", 0) for r in all_detections if r.get("kept"))

    logger.event("run_done", {
        "files_total": len(all_files),
        "files_kept": files_kept,
        "records_total": records_total,
        "jsonl_dir": str(jsonl_dir),
    })

    print(
        f"Pipeline finished.\n"
        f"  jsonl: {jsonl_dir}\n"
        f"  evaluation: {eval_dir}\n"
        f"  total files: {len(all_files)}, kept (en): {files_kept}, records: {records_total}"
    )

    return {
        "files_total": len(all_files),
        "files_kept": files_kept,
        "records_total": records_total,
    }
