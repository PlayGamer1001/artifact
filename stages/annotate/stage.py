# -*- coding: utf-8 -*-
"""Annotate stage: L1/L2 classification pipeline.

Orchestrates two-phase annotation:
  Phase 1 (L1): Classify each sentence into one or more Level-1 categories.
  Phase 2 (L2): For each L1 hit, route to Level-2 sub-categories.
Results are written to output/<target_dir>/annotate/answer/ as JSONL.
"""

from __future__ import annotations

import argparse
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from tqdm import tqdm

from .. import config
from ..clause_loader import Level1Category, load_clauses
from ..lib.jsonl_strict import (
    JsonlValidationError,
    handle_jsonl_validation_and_exit,
    load_done_line_index_set_from_answer_jsonl,
    load_stage1_l1_records_strict,
)
from ..lib.output import SortedJsonlWriter
from ..extract.postprocess import normalize_level1_level2
from ..lib.qwen_client import QwenLabelClient, sampling_from_mode
from ..lib.runtime_logger import RunLogger, append_run_error_log
from ..lib.sentence_reader import iter_sentences
from ..lib.util import chunked, should_skip_jsonl_path

SentenceRow = Tuple[int, str, List[str]]  # (line_index, sentence, heading_context)


# ---------------------------------------------------------------------------
# Helper functions (extracted from main.py)
# ---------------------------------------------------------------------------


def _load_source_file_list(path: Path | None) -> Set[str]:
    """Load a newline-separated list of source files from a text file."""
    if path is None or not Path(path).exists():
        return set()
    items: Set[str] = set()
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip().replace("\\", "/")
        if s:
            items.add(s)
    return items


def _append_source_file(path: Path | None, source_file: str) -> None:
    """Append a single source_file entry to a newline-separated tracking file."""
    if path is None:
        return
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{source_file}\n")


def _append_failed_source_file(path: Path | None, source_file: str) -> None:
    """Log a failed source file (same as _append_source_file, separate for clarity)."""
    _append_source_file(path, source_file)


def _collect_all_source_files(pp_jsonl_root: Path) -> List[str]:
    """Recursively collect all .jsonl file paths under pp_jsonl_root."""
    all_source_files: List[str] = []
    for jsonl_path in sorted(pp_jsonl_root.rglob("*.jsonl")):
        if should_skip_jsonl_path(jsonl_path):
            continue
        try:
            all_source_files.append(jsonl_path.relative_to(pp_jsonl_root).as_posix())
        except ValueError:
            all_source_files.append(jsonl_path.name)
    return all_source_files


def _collect_source_rows(
    pp_jsonl_root: Path,
    source_file: str,
    skip_empty: bool,
    min_length: int,
) -> List[SentenceRow]:
    """Load all sentence rows from a single source JSONL file."""
    rows: List[SentenceRow] = []
    for _sf, idx, sent, ctx in iter_sentences(
        pp_jsonl_root=pp_jsonl_root,
        skip_empty=skip_empty,
        min_length=min_length,
        include_source_files={source_file},
    ):
        rows.append((idx, sent, ctx))
    return rows


def _run_level1_rows(
    rows: List[SentenceRow],
    client: QwenLabelClient,
    categories: List[Level1Category],
    l1_concurrency: int,
    logger: RunLogger,
    run_error_log: Path,
    source_file: str,
) -> List[Dict[str, Any]]:
    """Run L1 classification on a batch of sentence rows with concurrency."""
    if not rows:
        return []

    indexed_rows = list(enumerate(rows))
    output_rows: List[Tuple[int, int, str, List[str], List[str]]] = []
    failed_rows: List[Tuple[int, int, str, List[str]]] = []

    def _run_one_item(item: Tuple[int, SentenceRow]) -> Tuple[int, int, str, List[str], List[str]]:
        row_pos, (line_idx, sent, ctx) = item
        one_l1 = client.classify_level1(sent, categories, heading_context=ctx)
        return (row_pos, line_idx, sent, ctx, one_l1)

    with ThreadPoolExecutor(max_workers=l1_concurrency) as executor:
        futures = {
            executor.submit(_run_one_item, item): item for item in indexed_rows
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="L1", leave=False):
            row_pos, (line_idx, sent, ctx) = futures[future]
            try:
                output_rows.append(future.result())
            except Exception as err:
                logger.error(
                    error_type="annotate_l1_row_failed",
                    message=str(err),
                    source_file=source_file,
                    line_index=line_idx,
                    traceback_text=traceback.format_exc(),
                )
                append_run_error_log(
                    run_error_log,
                    {
                        "stage": "annotate_L1_row",
                        "source_file": source_file,
                        "line_index": line_idx,
                        "error": str(err),
                    },
                )
                failed_rows.append((line_idx, sent, ctx))
                continue

    output_rows.sort(key=lambda x: x[0])
    records: List[Dict[str, Any]] = []
    for _row_pos, line_idx, sent, ctx, l1 in output_rows:
        records.append(
            {
                "line_index": line_idx,
                "sentence": sent,
                "heading_context": ctx,
                "level1": l1,
                "has_task_error": False,
            }
        )
    for line_idx, sent, ctx in failed_rows:
        records.append(
            {
                "line_index": line_idx,
                "sentence": sent,
                "heading_context": ctx,
                "level1": [],
                "has_task_error": True,
            }
        )
    return records


def _stage_file_path(root: Path, source_file: str) -> Path:
    """Map source_file to its corresponding stage1 output path."""
    return root / Path(source_file).with_suffix(".jsonl")


def _process_one_l1_record_for_l2(
    record: Dict[str, Any],
    name_to_category: Dict[str, Level1Category],
    client: QwenLabelClient,
) -> Dict[str, Any]:
    """For a single L1 record, run L2 classification for each hit L1 category."""
    line_index = int(record["line_index"])
    level1_list: List[str] = list(record["level1"])

    sentence = str(record["sentence"])
    heading_context = record.get("heading_context") or []

    # Build list of L1 categories that have L2 children
    l2_tasks: List[str] = []
    for l1 in dict.fromkeys(level1_list):
        if l1 == config.NO_LABEL:
            continue
        cat = name_to_category.get(l1)
        if not cat or not cat.children:
            continue
        l2_tasks.append(l1)

    # Run L2 classification for each L1
    level2_dict: Dict[str, List[str]] = {}
    if l2_tasks:
        for l1 in l2_tasks:
            cat = name_to_category.get(l1)
            if not cat or not cat.children:
                continue
            vals = client.classify_level2(
                sentence=sentence,
                level1_name=l1,
                level2_options=cat.children,
                heading_context=heading_context,
            )
            level2_dict[l1] = list(vals)

    # Normalize L1/L2 and decide whether to keep this record
    norm_l1, norm_l2 = normalize_level1_level2(level1_list, level2_dict)
    has_any_level2_hit = any(norm_l2.get(k) for k in norm_l2)

    out_record = None
    if has_any_level2_hit:
        out_record = {
            "source_file": str(record["source_file"]),
            "line_index": line_index,
            "sentence": sentence,
            "heading_context": heading_context,
            "level1": norm_l1,
            "level2": norm_l2,
        }

    return {
        "done_line_index": line_index,
        "output_record": out_record,
    }


# ---------------------------------------------------------------------------
# Main stage function
# ---------------------------------------------------------------------------


def run_annotate_stage(
    args: argparse.Namespace,
    logger: RunLogger,
    run_error_log: Path,
) -> Dict[str, int]:
    """Run the full annotate stage: L1 classification then L2 classification."""
    pp_jsonl = Path(args.pp_jsonl or config.PP_JSONL_ROOT)
    output_dir = Path(args.output_dir or config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_root = output_dir / args.target_dir
    target_root.mkdir(parents=True, exist_ok=True)

    run_root = output_dir / args.target_dir / "annotate"
    run_root.mkdir(parents=True, exist_ok=True)
    answer_root = run_root / "answer"
    answer_root.mkdir(parents=True, exist_ok=True)

    stage1_root = Path(args.l1_stage_dir or (run_root / config.L1_STAGE_DIRNAME))
    stage1_root.mkdir(parents=True, exist_ok=True)

    logger.paths.update(
        {
            "pp_jsonl": str(pp_jsonl),
            "target_dir_root": str(target_root),
            "run_errors_jsonl": str(run_error_log),
            "run_root": str(run_root),
            "answer_root": str(answer_root),
            "stage1_root": str(stage1_root),
        }
    )

    skip_empty = not args.no_skip_empty
    min_length = args.min_length if args.min_length is not None else config.MIN_SENTENCE_LENGTH
    l1_concurrency = args.l1_concurrency if args.l1_concurrency is not None else config.L1_CONCURRENCY
    l2_concurrency = args.l2_concurrency if args.l2_concurrency is not None else config.L2_CONCURRENCY
    files_per_batch = args.files_per_batch if args.files_per_batch is not None else config.FILES_PER_BATCH

    if l1_concurrency < 1 or l2_concurrency < 1:
        raise ValueError("l1-concurrency and l2-concurrency must both be >= 1")
    if files_per_batch < 1:
        raise ValueError("files-per-batch must be >= 1")

    l1_sampling = sampling_from_mode(args.l1_mode)
    l2_sampling = sampling_from_mode(args.l2_mode)

    empty_files_log = args.empty_files_log or (run_root / "empty_files.txt")
    empty_files_log_effective: Path | None = empty_files_log

    client = QwenLabelClient(
        base_url=args.base_url or config.QWEN_BASE_URL,
        model=args.model or config.QWEN_MODEL,
        l1_sampling=l1_sampling,
        l2_sampling=l2_sampling,
        l1_max_tokens=config.ANNOTATE_MAX_TOKENS,
        l2_max_tokens=config.ANNOTATE_MAX_TOKENS,
    )

    categories = load_clauses(json_path=args.clause_json or config.CLAUSE_JSON_PATH)
    name_to_category = {c.name: c for c in categories}

    all_source_files = _collect_all_source_files(pp_jsonl)
    if not all_source_files:
        logger.event("run_done", {"reason": "no_input_files"})
        print("No .jsonl files found. Exiting.")
        return {
            "l1_processed_files": 0,
            "total_l1_sentences": 0,
            "l2_processed_files": 0,
            "l2_written_records": 0,
        }

    skip_files = _load_source_file_list(args.skip_file_list)
    skip_files |= _load_source_file_list(empty_files_log_effective)

    candidate_source_files: List[str] = [sf for sf in all_source_files if sf not in skip_files]

    logger.bump("files_total", len(candidate_source_files))
    logger.event("batch_start", {"files_total": len(candidate_source_files), "stage": "annotate"})

    if not candidate_source_files:
        logger.event("run_done", {"reason": "no_candidate_files"})
        print("No candidate source files. Exiting.")
        return {
            "l1_processed_files": 0,
            "total_l1_sentences": 0,
            "l2_processed_files": 0,
            "l2_written_records": 0,
        }

    total_l1_sentences = 0
    l1_processed_files = 0
    l2_processed_files = 0
    l2_written_records = 0

    # -------------------------------------------------------------------------
    # Stage 1: L1 classification (fill in intermediate results)
    # -------------------------------------------------------------------------
    for file_batch in chunked(candidate_source_files, files_per_batch):
        for source_file in file_batch:
            stage_path = _stage_file_path(stage1_root, source_file)

            # Skip if all records already processed successfully
            if stage_path.exists():
                try:
                    existing = load_stage1_l1_records_strict(
                        stage_path, source_file=source_file, display_path=str(stage_path)
                    )
                    if all(not bool(r.get("has_task_error", False)) for r in existing):
                        continue
                except JsonlValidationError:
                    pass  # Proceed to reprocess

            logger.event("file_start", {"source_file": source_file, "phase": "L1"})
            try:
                try:
                    rows = _collect_source_rows(
                        pp_jsonl_root=pp_jsonl,
                        source_file=source_file,
                        skip_empty=skip_empty,
                        min_length=min_length,
                    )
                except JsonlValidationError as e:
                    handle_jsonl_validation_and_exit(
                        e,
                        run_error_log,
                        "annotate_L1_pp_jsonl",
                        {"source_file": source_file},
                    )
                l1_records = _run_level1_rows(
                    rows=rows,
                    client=client,
                    categories=categories,
                    l1_concurrency=l1_concurrency,
                    logger=logger,
                    run_error_log=run_error_log,
                    source_file=source_file,
                )
                for rec in l1_records:
                    rec["source_file"] = source_file

                writer = SortedJsonlWriter(stage_path)
                writer.append(l1_records)
                writer.flush()

                failed_count = sum(1 for r in l1_records if r.get("has_task_error", False))
                if failed_count:
                    logger.error(
                        error_type="annotate_l1_partial_failed",
                        message="L1 rows failed; written with has_task_error=True",
                        source_file=source_file,
                        payload={"failed_count": failed_count},
                    )

                total_l1_sentences += len(rows)
                logger.bump("sentences_l1_total", len(rows))
                l1_processed_files += 1
                logger.bump("l1_processed_files", 1)
                logger.event("file_done", {"source_file": source_file, "phase": "L1", "rows": len(rows)})
            except Exception as err:
                _append_failed_source_file(args.failed_files_log, source_file)
                tb = traceback.format_exc()
                payload = {
                    "stage": "annotate_L1",
                    "source_file": source_file,
                    "error": str(err),
                    "traceback": tb,
                }
                logger.error(
                    error_type="annotate_l1_failed",
                    message=str(err),
                    traceback_text=tb,
                    source_file=source_file,
                )
                append_run_error_log(run_error_log, payload)
                print(f"[ERROR] Stage aborted. stage=L1 source_file={source_file} error={err}")
                raise SystemExit(1)

    # -------------------------------------------------------------------------
    # Stage 2: L2 classification (read stage1, write final results)
    # -------------------------------------------------------------------------
    for file_batch in chunked(candidate_source_files, files_per_batch):
        for source_file in file_batch:
            stage_path = _stage_file_path(stage1_root, source_file)
            if not stage_path.exists():
                continue

            out_path = _stage_file_path(answer_root, source_file)

            # Load L2-done line_indices from answer JSONL (skip where has_task_error == False)
            done_line_idx: Set[int] = set()
            if out_path.exists():
                try:
                    done_line_idx = load_done_line_index_set_from_answer_jsonl(
                        out_path, display_path=str(out_path)
                    )
                except JsonlValidationError:
                    pass

            # Load stage1 records, skip rows where L1 failed (has_task_error == True)
            try:
                l1_records = load_stage1_l1_records_strict(
                    stage_path, source_file=source_file, display_path=str(stage_path)
                )
            except JsonlValidationError as e:
                handle_jsonl_validation_and_exit(
                    e,
                    run_error_log,
                    "annotate_L2_stage1",
                    {"source_file": source_file},
                )

            pending_l1_records = [
                rec for rec in l1_records
                if int(rec["line_index"]) not in done_line_idx and not bool(rec.get("has_task_error", False))
            ]

            if not pending_l1_records:
                l2_processed_files += 1
                logger.bump("l2_processed_files", 1)
                continue

            logger.event("file_start", {"source_file": source_file, "phase": "L2"})
            buffered_output_records: List[Dict[str, Any]] = []

            def _worker(rec: Dict[str, Any]) -> Dict[str, Any]:
                return _process_one_l1_record_for_l2(
                    record=rec,
                    name_to_category=name_to_category,
                    client=client,
                )

            try:
                with ThreadPoolExecutor(max_workers=l2_concurrency) as executor:
                    futures = {executor.submit(_worker, rec): rec for rec in pending_l1_records}
                    for future in tqdm(as_completed(futures), total=len(futures), desc="L2", leave=False):
                        rec = futures[future]
                        try:
                            one = future.result()
                        except Exception as err:
                            line_index = int(rec.get("line_index", -1))
                            buffered_output_records.append({
                                "source_file": source_file,
                                "line_index": line_index,
                                "sentence": rec.get("sentence", ""),
                                "heading_context": rec.get("heading_context", []),
                                "level1": rec.get("level1", []),
                                "level2": {},
                                "has_task_error": True,
                            })
                            logger.error(
                                error_type="annotate_l2_row_failed",
                                message=str(err),
                                traceback_text=traceback.format_exc(),
                                source_file=source_file,
                                line_index=line_index,
                            )
                            append_run_error_log(
                                run_error_log,
                                {
                                    "stage": "annotate_L2_row",
                                    "source_file": source_file,
                                    "line_index": line_index,
                                    "error": str(err),
                                },
                            )
                            continue

                        output_record = one["output_record"]
                        if output_record is not None:
                            output_record["has_task_error"] = False
                            buffered_output_records.append(output_record)
                            l2_written_records += 1
                            logger.bump("l2_written_records", 1)
                        else:
                            # No L2 hit, write success record
                            buffered_output_records.append({
                                "source_file": source_file,
                                "line_index": int(rec["line_index"]),
                                "sentence": rec.get("sentence", ""),
                                "heading_context": rec.get("heading_context", []),
                                "level1": rec.get("level1", []),
                                "level2": {},
                                "has_task_error": False,
                            })

                if buffered_output_records:
                    writer = SortedJsonlWriter(out_path)
                    writer.append(buffered_output_records)
                    writer.flush()

                l2_processed_files += 1
                logger.bump("l2_processed_files", 1)
                logger.event(
                    "file_done",
                    {"source_file": source_file, "phase": "L2", "records_written": len(buffered_output_records)},
                )
            except Exception as err:
                _append_failed_source_file(args.failed_files_log, source_file)
                tb = traceback.format_exc()
                payload = {
                    "stage": "annotate_L2",
                    "source_file": source_file,
                    "error": str(err),
                    "traceback": tb,
                }
                logger.error(
                    error_type="annotate_l2_failed",
                    message=str(err),
                    traceback_text=tb,
                    source_file=source_file,
                )
                append_run_error_log(run_error_log, payload)
                print(f"[ERROR] Stage aborted. stage=L2 source_file={source_file} error={err}")
                raise SystemExit(1)
            logger.event(
                "progress",
                {
                    "l1_processed_files": l1_processed_files,
                    "l2_processed_files": l2_processed_files,
                    "l2_written_records": l2_written_records,
                },
            )

    logger.event(
        "run_done",
        {
            "l1_processed_files": l1_processed_files,
            "total_l1_sentences": total_l1_sentences,
            "l2_processed_files": l2_processed_files,
            "l2_written_records": l2_written_records,
            "run_root": str(run_root),
        },
    )
    print(
        f"Done: L1 processed {l1_processed_files} files / {total_l1_sentences} sentences; "
        f"L2 processed {l2_processed_files} files, wrote {l2_written_records} results; "
        f"output: {answer_root}"
    )
    return {
        "l1_processed_files": l1_processed_files,
        "total_l1_sentences": total_l1_sentences,
        "l2_processed_files": l2_processed_files,
        "l2_written_records": l2_written_records,
    }
