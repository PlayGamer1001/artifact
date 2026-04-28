# -*- coding: utf-8 -*-
"""Normalize stage: normalize data/source/recipient/actor/purpose fields in extract results to taxonomy labels."""

from __future__ import annotations

import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from tqdm import tqdm

from .. import config
from ..lib.jsonl_strict import (
    JsonlValidationError,
    JsonlLineIssue,
    handle_jsonl_validation_and_exit,
    load_jsonl_dict_rows_strict,
    load_jsonl_records_with_int_line_index_strict,
)
from .labels import normalize_label
from .prompts import (
    NORMALIZE_SYSTEM_PROMPT,
    build_normalize_user_prompt,
)
from ..lib.output import (
    SortedJsonlWriter,
    SortedPairsJsonlWriter,
)
from ..lib.qwen_client import OpenAIChatClient, sampling_from_mode
from ..lib.runtime_logger import RunLogger, append_run_error_log
from ..lib.util import chunked, should_skip_jsonl_path

NORMALIZE_FIELDS = ("data", "source", "recipient", "actor", "purpose")

# Each extract file gets a same-named subdirectory in normalize/ containing two jsonl files
NORMALIZATION_PAIRS_JSONL = "normalization_pairs.jsonl"
EXTRACTION_NORMALIZED_FIELDS_JSONL = "extraction_with_normalized_fields.jsonl"

TaskKey = Tuple[str, int, str, int]  # ppi_key, obj_idx, field, elem_idx
CacheKey = Tuple[str, str]  # (kind, normalized_text)
ENTITY_FIELDS = frozenset({"source", "recipient", "actor"})


def _normalize_cache_text(text: str) -> str:
    """Normalize extracted text for cache key stability."""
    return " ".join(str(text or "").strip().lower().split())


def _cache_kind_for_field(field: str) -> str:
    """Map source/recipient/actor into one shared entity cache namespace."""
    if field in ENTITY_FIELDS:
        return "entity"
    return field


def _cache_key_for_task(field: str, text: str) -> CacheKey:
    return (_cache_kind_for_field(field), _normalize_cache_text(text))


def _collect_extract_files(extract_root: Path) -> List[Path]:
    """Recursively collect all .jsonl files under ``extract/`` (same layout as extract output)."""
    if not extract_root.exists():
        return []
    found = sorted(extract_root.rglob("*.jsonl"))
    return [p for p in found if p.is_file() and not should_skip_jsonl_path(p)]


def _load_existing_line_indices(rows_jsonl: Path) -> Set[int]:
    """Line indices already seen in normalize stage output (only for this file in this stage; unrelated to annotate/extract)."""
    if not rows_jsonl.exists():
        return set()
    rows = load_jsonl_records_with_int_line_index_strict(
        rows_jsonl, display_path=str(rows_jsonl)
    )
    return {r["line_index"] for r in rows if not bool(r.get("has_task_error", False))}


def _iter_normalize_tasks(
    extraction: Dict[str, Any],
) -> List[Tuple[TaskKey, str]]:
    tasks: List[Tuple[TaskKey, str]] = []
    for ppi_key, items in extraction.items():
        if not isinstance(items, list):
            continue
        for obj_idx, obj in enumerate(items):
            if not isinstance(obj, dict):
                continue
            for field in NORMALIZE_FIELDS:
                if field not in obj:
                    continue
                val = obj[field]
                if not isinstance(val, list):
                    continue
                for elem_idx, text in enumerate(val):
                    t = str(text).strip() if text is not None else ""
                    if not t:
                        continue
                    tasks.append(((ppi_key, obj_idx, field, elem_idx), t))
    return tasks


def _iter_all_normalize_field_slots(
    extraction: Dict[str, Any],
) -> List[Tuple[TaskKey, str]]:
    """One-to-one with extraction's normalize field slots (including empty strings), used to write normalization_pairs."""
    slots: List[Tuple[TaskKey, str]] = []
    for ppi_key, items in extraction.items():
        if not isinstance(items, list):
            continue
        for obj_idx, obj in enumerate(items):
            if not isinstance(obj, dict):
                continue
            for field in NORMALIZE_FIELDS:
                if field not in obj:
                    continue
                val = obj[field]
                if not isinstance(val, list):
                    continue
                for elem_idx, text in enumerate(val):
                    t = str(text).strip() if text is not None else ""
                    slots.append(((ppi_key, obj_idx, field, elem_idx), t))
    return slots


def _build_extraction_with_label_lists(
    extraction: Dict[str, Any],
    labels_by_key: Dict[TaskKey, str],
) -> Dict[str, Any]:
    """Same shape as extract's extraction, but normalize-field list elements are normalized label strings."""
    out: Dict[str, Any] = {}
    for ppi_key, items in extraction.items():
        if not isinstance(items, list):
            continue
        out_list: List[Dict[str, Any]] = []
        for obj_idx, obj in enumerate(items):
            if not isinstance(obj, dict):
                out_list.append({})
                continue
            out_obj: Dict[str, Any] = {}
            for k, v in obj.items():
                if k in NORMALIZE_FIELDS:
                    continue
                out_obj[k] = v
            for field in NORMALIZE_FIELDS:
                if field not in obj or not isinstance(obj[field], list):
                    continue
                labels: List[str] = []
                for elem_idx, raw in enumerate(obj[field]):
                    text = str(raw).strip() if raw is not None else ""
                    key = (ppi_key, obj_idx, field, elem_idx)
                    if text:
                        lab = labels_by_key.get(key, "")
                    else:
                        lab = ""
                    labels.append(lab)
                out_obj[field] = labels
            out_list.append(out_obj)
        out[ppi_key] = out_list
    return out


def _run_one_task(
    client: OpenAIChatClient,
    sentence: str,
    field: str,
    text: str,
) -> Tuple[str, bool, str]:
    user = build_normalize_user_prompt(field, sentence, text)
    raw = client.chat(NORMALIZE_SYSTEM_PROMPT, user)
    label, matched = normalize_label(field, raw)
    return label, matched, raw


def run_normalize_stage(
    output_dir: Path,
    target_dir_name: str,
    base_url: str,
    model: str,
    mode: str,
    concurrency: int,
    files_per_batch: int,
    logger: RunLogger,
    run_error_log: Path,
) -> Dict[str, int]:
    if concurrency < 1 or files_per_batch < 1:
        raise ValueError("normalize-concurrency and normalize-files-per-batch must both be >= 1")

    extract_root = output_dir / target_dir_name / "extract"
    normalize_root = output_dir / target_dir_name / "normalize"
    normalize_root.mkdir(parents=True, exist_ok=True)

    files = _collect_extract_files(extract_root)
    logger.bump("files_total", len(files))
    logger.event(
        "batch_start",
        {
            "extract_root": str(extract_root),
            "normalize_root": str(normalize_root),
            "files_total": len(files),
            "normalize_concurrency": concurrency,
            "normalize_files_per_batch": files_per_batch,
            "normalize_mode": mode,
        },
    )
    if not files:
        logger.event("run_done", {"reason": "no_input_files"})
        return {"files_total": 0, "files_processed": 0, "records_written": 0}

    sampling = sampling_from_mode(mode)
    client = OpenAIChatClient(
        base_url=base_url,
        model=model,
        sampling=sampling,
        max_tokens=config.NORMALIZE_MAX_TOKENS,
    )
    # Global per-run cache: repeated extracted text will be normalized only once.
    normalize_cache: Dict[CacheKey, Tuple[str, bool]] = {}
    files_processed = 0
    records_written = 0

    for file_batch in chunked(files, files_per_batch):
        logger.event("batch_start", {"batch_files": len(file_batch)})
        for in_file in file_batch:
            rel_path = in_file.relative_to(extract_root)
            out_dir = normalize_root / rel_path.parent / rel_path.stem
            rows_jsonl = out_dir / EXTRACTION_NORMALIZED_FIELDS_JSONL
            pairs_jsonl = out_dir / NORMALIZATION_PAIRS_JSONL
            logger.event("file_start", {"file": str(rel_path), "out_dir": str(out_dir)})
            file_written = 0
            writer_rows = SortedJsonlWriter(rows_jsonl)
            writer_pairs = SortedPairsJsonlWriter(pairs_jsonl)
            try:
                try:
                    done_li = _load_existing_line_indices(rows_jsonl)
                except JsonlValidationError as e:
                    handle_jsonl_validation_and_exit(
                        e,
                        run_error_log,
                        "normalize",
                        {"normalize_rows_jsonl": str(rows_jsonl)},
                    )
                pending_rows: List[Dict[str, Any]] = []
                pending_pairs: List[Dict[str, Any]] = []
                try:
                    loaded = load_jsonl_dict_rows_strict(
                        in_file, display_path=str(rel_path)
                    )
                except JsonlValidationError as e:
                    handle_jsonl_validation_and_exit(
                        e,
                        run_error_log,
                        "normalize",
                        {"extract_subpath": str(rel_path)},
                    )
                schema_issues: List[JsonlLineIssue] = []
                for phys_line, rec in loaded:
                    li = rec.get("line_index")
                    if type(li) is not int:
                        schema_issues.append(
                            JsonlLineIssue(
                                str(rel_path),
                                phys_line,
                                "invalid_line_index",
                                "extract rows must contain int line_index",
                                "",
                            )
                        )
                if schema_issues:
                    handle_jsonl_validation_and_exit(
                        JsonlValidationError(schema_issues),
                        run_error_log,
                        "normalize",
                        {"extract_subpath": str(rel_path)},
                    )
                for _phys_line, rec in loaded:
                    li = rec["line_index"]
                    if li in done_li:
                        continue

                    extraction = rec.get("extraction")
                    if not isinstance(extraction, dict):
                        row_rec = {
                            "source_file": rec.get("source_file"),
                            "line_index": rec.get("line_index"),
                            "sentence": rec.get("sentence"),
                            "extraction": {},
                            "has_task_error": False,
                        }
                        pending_rows.append(row_rec)
                        file_written += 1
                        records_written += 1
                        logger.bump("records_written", 1)
                        continue

                    task_pairs = _iter_normalize_tasks(extraction)
                    labels_by_key: Dict[TaskKey, str] = {}
                    has_task_error = False
                    if task_pairs:
                        uncached_groups: Dict[CacheKey, Dict[str, Any]] = {}
                        for key, text in task_pairs:
                            _ppi, _oi, field, _ei = key
                            ck = _cache_key_for_task(field, text)
                            cached = normalize_cache.get(ck)
                            if cached is not None:
                                labels_by_key[key] = cached[0]
                                logger.bump("normalize_cache_hits", 1)
                                continue
                            group = uncached_groups.get(ck)
                            if group is None:
                                uncached_groups[ck] = {
                                    "field": field,
                                    "text": text,
                                    "keys": [key],
                                }
                            else:
                                group["keys"].append(key)

                        if uncached_groups:
                            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                                future_map = {}
                                sentence = str(rec.get("sentence", ""))
                                for ck, group in uncached_groups.items():
                                    fut = executor.submit(
                                        _run_one_task,
                                        client,
                                        sentence,
                                        str(group["field"]),
                                        str(group["text"]),
                                    )
                                    future_map[fut] = ck
                                    logger.bump("normalize_cache_misses", 1)

                                for fut in tqdm(
                                    as_completed(future_map),
                                    total=len(future_map),
                                    desc="normalize",
                                    leave=False,
                                ):
                                    ck = future_map[fut]
                                    group = uncached_groups[ck]
                                    field = str(group["field"])
                                    keys = group["keys"]
                                    logger.bump("normalize_requests_total", 1)
                                    try:
                                        label, matched, raw = fut.result()
                                    except Exception as err:
                                        has_task_error = True
                                        logger.error(
                                            error_type="normalize_task_failed",
                                            message=str(err),
                                            source_file=str(rel_path),
                                            line_index=li,
                                            traceback_text=traceback.format_exc(),
                                            payload={"field": field},
                                        )
                                        append_run_error_log(
                                            run_error_log,
                                            {
                                                "stage": "normalize",
                                                "error": str(err),
                                                "source_file": str(rel_path),
                                                "line_index": li,
                                                "field": field,
                                            },
                                        )
                                        for key in keys:
                                            labels_by_key[key] = ""
                                        continue

                                    normalize_cache[ck] = (label, matched)
                                    for key in keys:
                                        labels_by_key[key] = label
                                    if not matched and label:
                                        logger.error(
                                            error_type="normalize_label_unmatched",
                                            message="model output not in taxonomy",
                                            source_file=str(rel_path),
                                            line_index=li,
                                            payload={
                                                "field": field,
                                                "raw_output": raw[:500],
                                                "canonical": label,
                                            },
                                        )

                    ext_labels = _build_extraction_with_label_lists(
                        extraction, labels_by_key
                    )
                    row_rec = {
                        "source_file": rec.get("source_file"),
                        "line_index": rec.get("line_index"),
                        "sentence": rec.get("sentence"),
                        "extraction": ext_labels,
                        "has_task_error": has_task_error,
                    }
                    pending_rows.append(row_rec)

                    src_name = rec.get("source_file", "")
                    li_val = rec.get("line_index")
                    for key, text in _iter_all_normalize_field_slots(extraction):
                        ppi_key, obj_idx, field, elem_idx = key
                        t = str(text).strip() if text is not None else ""
                        if t:
                            lab = labels_by_key.get(key, "")
                        else:
                            lab = ""
                        pending_pairs.append(
                            {
                                "source_file": src_name,
                                "line_index": li_val,
                                "ppi_key": ppi_key,
                                "object_index": obj_idx,
                                "field": field,
                                "element_index": elem_idx,
                                "extracted_text": t,
                                "normalized_label": lab,
                            }
                        )

                    file_written += 1
                    records_written += 1
                    logger.bump("records_written", 1)

                if pending_rows or pending_pairs:
                    out_dir.mkdir(parents=True, exist_ok=True)
                if pending_pairs:
                    writer_pairs.append(pending_pairs)
                if pending_rows:
                    writer_rows.append(pending_rows)
            except SystemExit:
                raise
            except Exception as err:
                tb = traceback.format_exc()
                logger.error(
                    error_type="file_process_failed",
                    message=str(err),
                    source_file=str(rel_path),
                    traceback_text=tb,
                )
                append_run_error_log(
                    run_error_log,
                    {
                        "stage": "normalize",
                        "error": str(err),
                        "traceback": tb,
                        "source_file": str(rel_path),
                    },
                )
                raise SystemExit(1) from err
            finally:
                writer_rows.flush()
                writer_pairs.flush()
            files_processed += 1
            logger.bump("files_processed", 1)
            logger.event(
                "file_done",
                {"file": str(rel_path), "records_written": file_written},
            )
            logger.event(
                "progress",
                {"files_processed": files_processed, "files_total": len(files)},
            )

    logger.event(
        "run_done",
        {
            "files_processed": files_processed,
            "records_written": records_written,
            "normalize_root": str(normalize_root),
        },
    )
    return {
        "files_total": len(files),
        "files_processed": files_processed,
        "records_written": records_written,
    }
