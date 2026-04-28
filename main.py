# -*- coding: utf-8 -*-
"""CLI entry point for the Auditing_Pipeline.

Run with: python main.py --target-dir <name> [--process] [--annotate] [--extract] [--normalize]

Stage order (when multiple are specified): process -> annotate -> extract -> normalize
"""

from __future__ import annotations

import argparse
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import stages.config as config
from stages.annotate.stage import run_annotate_stage
from stages.extract.stage import run_extract_stage
from stages.normalize.stage import run_normalize_stage
from stages.lib.runtime_logger import RunLogger, append_run_error_log, build_run_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate, extract, and normalize privacy policy sentences using LLM classification.",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        required=True,
        help="Output root directory name (under <output-dir>/). "
             "Contains annotate/, extract/, normalize/ subdirectories.",
    )

    parser.add_argument(
        "--process",
        action="store_true",
        help="Run process stage: convert raw HTML/DOCX/PDF/TXT to JSONL sentences.",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Run annotate stage (L1/L2 classification), writes to <output-dir>/<target-dir>/annotate/",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Run extract stage (structured extraction), reads annotate/answer/, writes to <output-dir>/<target-dir>/extract/",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Run normalize stage (taxonomy normalization), reads extract/, writes to <output-dir>/<target-dir>/normalize/",
    )

    parser.add_argument(
        "--pp-jsonl",
        type=Path,
        default=None,
        help=f"PP_jsonl root directory, default: {config.PP_JSONL_ROOT}",
    )
    parser.add_argument(
        "--clause-json",
        type=Path,
        default=None,
        help=f"Clause labels JSON path, default: {config.CLAUSE_JSON_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Output base directory, default: {config.OUTPUT_DIR}",
    )
    parser.add_argument(
        "--process-input-dir",
        type=Path,
        default=None,
        help=f"Process stage input directory (raw files), default: <project-root>/data/input",
    )

    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help=f"Qwen API base URL, default: {config.QWEN_BASE_URL}",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Model name, default: {config.QWEN_MODEL}",
    )

    parser.add_argument(
        "--l1-stage-dir",
        type=Path,
        default=None,
        help="L1 intermediate results directory, default: <run_root>/_stage1",
    )
    parser.add_argument(
        "--no-skip-empty",
        action="store_true",
        help="Do not skip empty lines in input (default: skip)",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=None,
        help=f"Minimum sentence length (chars), default: {config.MIN_SENTENCE_LENGTH}",
    )
    parser.add_argument(
        "--l1-mode",
        type=str,
        default="nothinking",
        choices=("thinking", "nothinking"),
        help="L1 classification mode, default: nothinking",
    )
    parser.add_argument(
        "--l2-mode",
        type=str,
        default="nothinking",
        choices=("thinking", "nothinking"),
        help="L2 classification mode, default: nothinking",
    )
    parser.add_argument(
        "--l1-concurrency",
        type=int,
        default=None,
        help=f"L1 concurrency, default: {config.L1_CONCURRENCY}",
    )
    parser.add_argument(
        "--l2-concurrency",
        type=int,
        default=None,
        help=f"L2 concurrency, default: {config.L2_CONCURRENCY}",
    )
    parser.add_argument(
        "--files-per-batch",
        type=int,
        default=None,
        help=f"Files per batch, default: {config.FILES_PER_BATCH}",
    )
    parser.add_argument(
        "--skip-file-list",
        type=Path,
        default=None,
        help="Text file with source_file entries to skip (one per line)",
    )
    parser.add_argument(
        "--empty-files-log",
        type=Path,
        default=None,
        help="Log of source files that produced no L2 hits; auto-skipped on re-run",
    )
    parser.add_argument(
        "--failed-files-log",
        type=Path,
        default=None,
        help="Log path for files that failed with errors",
    )

    parser.add_argument(
        "--extract-mode",
        type=str,
        default=config.EXTRACT_MODE,
        choices=("thinking", "nothinking"),
        help=f"Extract stage mode, default: {config.EXTRACT_MODE}",
    )
    parser.add_argument(
        "--extract-concurrency",
        type=int,
        default=None,
        help=f"Extract concurrency, default: {config.EXTRACT_CONCURRENCY}",
    )
    parser.add_argument(
        "--extract-files-per-batch",
        type=int,
        default=None,
        help=f"Extract files per batch, default: {config.EXTRACT_FILES_PER_BATCH}",
    )

    parser.add_argument(
        "--normalize-mode",
        type=str,
        default=config.NORMALIZE_MODE,
        choices=("thinking", "nothinking"),
        help=f"Normalize stage mode, default: {config.NORMALIZE_MODE}",
    )
    parser.add_argument(
        "--normalize-concurrency",
        type=int,
        default=None,
        help=f"Normalize concurrency, default: {config.NORMALIZE_CONCURRENCY}",
    )
    parser.add_argument(
        "--normalize-files-per-batch",
        type=int,
        default=None,
        help=f"Normalize files per batch, default: {config.NORMALIZE_FILES_PER_BATCH}",
    )

    parser.add_argument(
        "--process-concurrency",
        type=int,
        default=config.PROCESS_CONCURRENCY,
        help=f"Process stage concurrency, default: {config.PROCESS_CONCURRENCY}",
    )
    parser.add_argument(
        "--process-files-per-batch",
        type=int,
        default=config.PROCESS_FILES_PER_BATCH,
        help=f"Process files per batch, default: {config.PROCESS_FILES_PER_BATCH}",
    )

    parser.add_argument(
        "--log-root",
        type=Path,
        default=None,
        help=f"Log root directory, default: <output-dir>/_logs",
    )
    parser.add_argument(
        "--error-log",
        type=Path,
        default=None,
        help=f"Fatal errors JSONL path, default: <output-dir>/<target-dir>/run_errors.jsonl",
    )

    return parser.parse_args()


def _build_params_snapshot(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "target_dir": args.target_dir,
        "process": bool(args.process),
        "annotate": bool(args.annotate),
        "extract": bool(args.extract),
        "normalize": bool(args.normalize),
        "model": args.model or config.QWEN_MODEL,
        "base_url": args.base_url or config.QWEN_BASE_URL,
        "pp_jsonl": str(args.pp_jsonl) if args.pp_jsonl else str(config.PP_JSONL_ROOT),
        "output_dir": str(args.output_dir) if args.output_dir else str(config.OUTPUT_DIR),
        "process_input_dir": str(args.process_input_dir) if args.process_input_dir else "",
        "l1_mode": args.l1_mode,
        "l2_mode": args.l2_mode,
        "extract_mode": args.extract_mode,
        "extract_concurrency": args.extract_concurrency,
        "extract_files_per_batch": args.extract_files_per_batch,
        "normalize_mode": args.normalize_mode,
        "normalize_concurrency": args.normalize_concurrency,
        "normalize_files_per_batch": args.normalize_files_per_batch,
        "process_concurrency": args.process_concurrency,
        "process_files_per_batch": args.process_files_per_batch,
        "l1_concurrency": args.l1_concurrency,
        "l2_concurrency": args.l2_concurrency,
        "files_per_batch": args.files_per_batch,
    }


def main() -> None:
    args = parse_args()
    if not any([args.process, args.annotate, args.extract, args.normalize]):
        raise ValueError("At least one of --process, --annotate, --extract, or --normalize is required")

    output_dir = Path(args.output_dir or config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_root = output_dir / args.target_dir
    target_root.mkdir(parents=True, exist_ok=True)
    run_error_log = Path(args.error_log) if args.error_log else (target_root / "run_errors.jsonl")
    log_root = Path(args.log_root or (output_dir / "_logs"))
    base_paths = {
        "output_dir": str(output_dir),
        "log_root": str(log_root),
        "target_dir_root": str(target_root),
        "run_errors_jsonl": str(run_error_log),
    }
    ts_start = datetime.now(timezone.utc).isoformat()

    # --- Process stage ---
    if args.process:
        from stages.process.stage import run_process_stage
        run_id = build_run_id("process")
        logger = RunLogger(
            log_root=log_root,
            stage="process",
            run_id=run_id,
            params=_build_params_snapshot(args),
            paths=dict(base_paths),
        )
        logger.event("run_start", {"ts_utc": ts_start})
        try:
            run_process_stage(
                output_dir=output_dir,
                target_dir_name=args.target_dir,
                base_url="",
                model="",
                mode="",
                concurrency=args.process_concurrency,
                files_per_batch=args.process_files_per_batch,
                logger=logger,
                run_error_log=run_error_log,
                input_dir=args.process_input_dir,
            )
        except SystemExit:
            logger.set_status("failed")
            raise
        except Exception as err:
            logger.set_status("failed")
            tb = traceback.format_exc()
            logger.error(
                error_type="run_failed",
                message=str(err),
                traceback_text=tb,
            )
            append_run_error_log(
                run_error_log,
                {"stage": "process", "error": str(err), "traceback": tb},
            )
            raise
        finally:
            logger.write_summary()

    # --- Annotate stage ---
    if args.annotate:
        run_id = build_run_id("annotate")
        logger = RunLogger(
            log_root=log_root,
            stage="annotate",
            run_id=run_id,
            params=_build_params_snapshot(args),
            paths=dict(base_paths),
        )
        logger.event("run_start", {"ts_utc": ts_start})
        try:
            run_annotate_stage(args, logger, run_error_log)
        except SystemExit:
            logger.set_status("failed")
            raise
        except Exception as err:
            logger.set_status("failed")
            tb = traceback.format_exc()
            logger.error(
                error_type="run_failed",
                message=str(err),
                traceback_text=tb,
            )
            append_run_error_log(
                run_error_log,
                {"stage": "annotate", "error": str(err), "traceback": tb},
            )
            raise
        finally:
            logger.write_summary()

    # --- Extract stage ---
    if args.extract:
        run_id = build_run_id("extract")
        logger = RunLogger(
            log_root=log_root,
            stage="extract",
            run_id=run_id,
            params=_build_params_snapshot(args),
            paths=dict(base_paths),
        )
        logger.event("run_start", {"ts_utc": datetime.now(timezone.utc).isoformat()})
        try:
            run_extract_stage(
                output_dir=output_dir,
                target_dir_name=args.target_dir,
                base_url=args.base_url or config.QWEN_BASE_URL,
                model=args.model or config.QWEN_MODEL,
                mode=args.extract_mode,
                concurrency=args.extract_concurrency if args.extract_concurrency is not None else config.EXTRACT_CONCURRENCY,
                files_per_batch=(
                    args.extract_files_per_batch
                    if args.extract_files_per_batch is not None
                    else config.EXTRACT_FILES_PER_BATCH
                ),
                logger=logger,
                run_error_log=run_error_log,
            )
        except SystemExit:
            logger.set_status("failed")
            raise
        except Exception as err:
            logger.set_status("failed")
            tb = traceback.format_exc()
            logger.error(
                error_type="run_failed",
                message=str(err),
                traceback_text=tb,
            )
            append_run_error_log(
                run_error_log,
                {"stage": "extract", "error": str(err), "traceback": tb},
            )
            raise
        finally:
            logger.write_summary()

    # --- Normalize stage ---
    if args.normalize:
        run_id = build_run_id("normalize")
        logger = RunLogger(
            log_root=log_root,
            stage="normalize",
            run_id=run_id,
            params=_build_params_snapshot(args),
            paths=dict(base_paths),
        )
        logger.event("run_start", {"ts_utc": datetime.now(timezone.utc).isoformat()})
        try:
            run_normalize_stage(
                output_dir=output_dir,
                target_dir_name=args.target_dir,
                base_url=args.base_url or config.QWEN_BASE_URL,
                model=args.model or config.QWEN_MODEL,
                mode=args.normalize_mode,
                concurrency=(
                    args.normalize_concurrency
                    if args.normalize_concurrency is not None
                    else config.NORMALIZE_CONCURRENCY
                ),
                files_per_batch=(
                    args.normalize_files_per_batch
                    if args.normalize_files_per_batch is not None
                    else config.NORMALIZE_FILES_PER_BATCH
                ),
                logger=logger,
                run_error_log=run_error_log,
            )
        except SystemExit:
            logger.set_status("failed")
            raise
        except Exception as err:
            logger.set_status("failed")
            tb = traceback.format_exc()
            logger.error(
                error_type="run_failed",
                message=str(err),
                traceback_text=tb,
            )
            append_run_error_log(
                run_error_log,
                {"stage": "normalize", "error": str(err), "traceback": tb},
            )
            raise
        finally:
            logger.write_summary()


if __name__ == "__main__":
    main()
