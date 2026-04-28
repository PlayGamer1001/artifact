# -*- coding: utf-8 -*-
"""Extract stage: filter annotated results by target labels and extract with single-request-per-task concurrency."""

from __future__ import annotations

import copy
import json
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Set, Tuple

from tqdm import tqdm

from .. import config
from .prompts import (
    SIX_LABEL_SYSTEM_PROMPT,
    build_six_user_prompt,
)
from ..lib.jsonl_strict import (
    JsonlValidationError,
    JsonlLineIssue,
    handle_jsonl_validation_and_exit,
    load_jsonl_dict_rows_strict,
    load_jsonl_records_with_int_line_index_strict,
)
from ..lib.output import SortedJsonlWriter
from ..lib.qwen_client import OpenAIChatClient, sampling_from_mode
from ..lib.resilience import parse_with_retries
from ..lib.runtime_logger import RunLogger, append_run_error_log
from ..lib.util import chunked, should_skip_jsonl_path

SIX_TARGET_LABELS: List[str] = [
    "Scope of PI Collection",
    "Sources of PI Collected",
    "Scope of PI Sold/Shared/Disclosed",
    "Third-Party Recipients of PI",
    "Purposes for PI Processing",
    "PI Retention Periods/Criteria",
]

DATE_ONLY_LABEL: str = "Last Updated Date/Effective Date"

SINGLE_LABEL_NAMES: Tuple[str, ...] = ()

SINGLE_LABELS: FrozenSet[str] = frozenset(SINGLE_LABEL_NAMES)

TARGET_LABELS: List[str] = [*SIX_TARGET_LABELS]

_RAW_PREVIEW_LEN = 2000


def _preview_raw_for_error(raw: str) -> str:
    s = raw if isinstance(raw, str) else repr(raw)
    if len(s) <= _RAW_PREVIEW_LEN:
        return s
    return s[:_RAW_PREVIEW_LEN] + "...(truncated)"


def _extract_dates_regex(sentence: str) -> List[Dict[str, str]]:
    """Extract dates from sentence using regex (regex-only path, no LLM fallback).

    Output items keep backward-compatible ``date`` field and add:
    - precision: day|month|year
    - method: regex
    - confidence: high|medium|low
    """
    from dateutil import parser as dateutil_parser

    text = str(sentence or "")
    results: List[Dict[str, str]] = []
    seen: Set[Tuple[str, str]] = set()

    # Tight context keywords for coarse year-only extraction.
    date_ctx_re = re.compile(
        r"\b(effective|effectivity|last\s+updated|updated|update|revised|reviewed|revision|modified|published|version)\b",
        re.IGNORECASE,
    )

    def _add(date_value: str, precision: str, confidence: str) -> None:
        key = (date_value, precision)
        if key in seen:
            return
        seen.add(key)
        results.append(
            {
                "date": date_value,
                "precision": precision,
                "method": "regex",
                "confidence": confidence,
            }
        )

    def _has_date_context_around(start: int, end: int, window: int = 48) -> bool:
        left = max(0, start - window)
        right = min(len(text), end + window)
        return date_ctx_re.search(text[left:right]) is not None

    def _has_richer_precision_for_year(year: str) -> bool:
        for rec in results:
            if rec.get("precision") in {"day", "month"} and str(rec.get("date", "")).startswith(f"{year}-"):
                return True
        return False

    # 1) Ordinal day + month + year  e.g. "12th August 2024"
    for m in re.finditer(r'\b(\d{1,2})(st|nd|rd|th)\s+([A-Za-z]{3,9})\s+(\d{4})\b', text, re.IGNORECASE):
        try:
            day = int(m.group(1))
            month_num = dateutil_parser.parse(f"{m.group(3)} 1 2000").strftime('%m')
            key = f"{m.group(4)}-{month_num}-{day:02d}"
            _add(key, "day", "high")
        except: pass

    # 2) "of" ordinal  e.g. "3rd of January, 2024"
    for m in re.finditer(r'\b(\d{1,2})(?:st|nd|rd|th)?\s+of\s+([A-Za-z]{3,9}),?\s+(\d{4})\b', text, re.IGNORECASE):
        try:
            day = int(m.group(1))
            month_num = dateutil_parser.parse(f"{m.group(2)} 1 2000").strftime('%m')
            key = f"{m.group(3)}-{month_num}-{day:02d}"
            _add(key, "day", "high")
        except: pass

    # 3) "Month DD, YYYY"  e.g. "August 12, 2024"
    for m in re.finditer(r'\b([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(\d{4})\b', text, re.IGNORECASE):
        try:
            month_num = dateutil_parser.parse(f"{m.group(1)} 1 2000").strftime('%m')
            key = f"{m.group(3)}-{month_num}-{int(m.group(2)):02d}"
            _add(key, "day", "high")
        except: pass

    # 4) DD-Mon-YYYY or DD.MM.YYYY  e.g. "12-Aug-2024" or "26.03.2025"
    for m in re.finditer(r'\b(\d{1,2})[-.](\d{1,2}|[A-Za-z]{3,9})[-.](\d{4})\b', text, re.IGNORECASE):
        try:
            first = int(m.group(1))
            second = m.group(2)
            year = m.group(3)
            if second.isdigit():
                second_int = int(second)
                confidence = "high"
                if first > 12:
                    key = f"{year}-{second_int:02d}-{first:02d}"
                elif second_int > 12:
                    key = f"{year}-{first:02d}-{second_int:02d}"
                else:
                    key = f"{year}-{second_int:02d}-{first:02d}"
                    confidence = "low"
            else:
                month_num = dateutil_parser.parse(f"{second} 1 2000").strftime('%m')
                key = f"{year}-{month_num}-{first:02d}"
                confidence = "high"
            _add(key, "day", confidence)
        except: pass

    # 5) YYYY-MM-DD
    for m in re.finditer(r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b', text):
        try:
            key = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
            _add(key, "day", "high")
        except: pass

    # 6) MM/DD/YYYY or DD/MM/YYYY
    for m in re.finditer(r'\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b', text):
        try:
            first, second, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if year < 100:
                year += 2000 if year < 50 else 1900
            confidence = "high"
            if first > 12:
                key = f"{year}-{second:02d}-{first:02d}"
            elif second > 12:
                key = f"{year}-{first:02d}-{second:02d}"
            else:
                key = f"{year}-{first:02d}-{second:02d}"
                confidence = "low"
            _add(key, "day", confidence)
        except: pass

    # 7) "Month YYYY"  e.g. "August 2025"
    for m in re.finditer(r'\b([A-Za-z]{3,9}),?\s+(\d{4})\b', text, re.IGNORECASE):
        if re.search(r'\d', m.group(1)):
            continue
        try:
            month_num = dateutil_parser.parse(f"{m.group(1)} 1 2000").strftime('%m')
            key = f"{m.group(2)}-{month_num}"
            _add(key, "month", "medium")
        except: pass

    # 8) Standalone "YYYY" year only (accept only with date-context keywords nearby)
    for m in re.finditer(r'\b(20\d{2})\b', text):
        if not _has_date_context_around(m.start(), m.end()):
            continue
        key = m.group(1)
        if _has_richer_precision_for_year(key):
            continue
        _add(key, "year", "low")

    # 9) Spaced year "202 6" -> "2026" (also needs date-context keywords)
    for m in re.finditer(r'\b(20\s*\d\s*\d)\b', text):
        if not _has_date_context_around(m.start(), m.end()):
            continue
        key = m.group(1).replace(' ', '')
        if _has_richer_precision_for_year(key):
            continue
        _add(key, "year", "low")

    return results


def _coerce_line_index(raw: Any) -> int | None:
    """Normalize line_index to int; accepts float that is a whole number (e.g. 1.0)."""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float) and raw.is_integer():
        return int(raw)
    return None


def _strip_code_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```\w*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    return s.strip()


def _extract_balanced(s: str, open_ch: str, close_ch: str) -> str | None:
    """Extract a balanced substring starting from the first open_ch (respects quotes and escapes)."""
    i = s.find(open_ch)
    if i < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    j = i
    while j < len(s):
        ch = s[j]
        if esc:
            esc = False
            j += 1
            continue
        if in_str:
            if ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            j += 1
            continue
        if ch == '"':
            in_str = True
            j += 1
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return s[i : j + 1]
        j += 1
    return None


def _parse_json_object(text: str) -> Dict[str, Any] | None:
    s = _strip_code_fence(text)
    candidates: List[str] = []
    t = s.strip()
    if t:
        candidates.append(t)
    bal = _extract_balanced(s, "{", "}")
    if bal and bal.strip() != t:
        candidates.append(bal.strip())
    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        g = m.group(0).strip()
        if g not in candidates:
            candidates.append(g)

    for c in candidates:
        if not c:
            continue
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _parse_json_array(text: str) -> List[Any] | None:
    s = _strip_code_fence(text)
    candidates: List[str] = []
    t = s.strip()
    if t:
        candidates.append(t)
    bal = _extract_balanced(s, "[", "]")
    if bal and bal.strip() != t:
        candidates.append(bal.strip())
    m = re.search(r"\[[\s\S]*\]", s)
    if m:
        g = m.group(0).strip()
        if g not in candidates:
            candidates.append(g)

    for c in candidates:
        if not c:
            continue
        try:
            arr = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(arr, list):
            return arr
    return None


def _collect_labeled_jsonl(annotate_root: Path) -> tuple[Path, List[Path]]:
    """Recursively collect all .jsonl files under ``annotate/answer`` (mirrors annotate output structure)."""
    answer_path = annotate_root / "answer"
    if not answer_path.exists():
        return answer_path, []
    found = sorted(answer_path.rglob("*.jsonl"))
    return answer_path, [p for p in found if p.is_file() and not should_skip_jsonl_path(p)]


def _target_labels_from_record(rec: Dict[str, Any]) -> List[str]:
    l2 = rec.get("level2")
    if not isinstance(l2, dict):
        return []
    labels: List[str] = []
    seen = set()
    for vals in l2.values():
        if not isinstance(vals, list):
            continue
        for raw in vals:
            name = str(raw)
            if name in TARGET_LABELS and name not in seen:
                seen.add(name)
                labels.append(name)
    return labels


def _has_date_only_label(rec: Dict[str, Any]) -> bool:
    """Return True if the record has Last Updated Date/Effective Date label (even if no other labels)."""
    l2 = rec.get("level2")
    if not isinstance(l2, dict):
        return False
    for vals in l2.values():
        if isinstance(vals, list) and DATE_ONLY_LABEL in [str(v) for v in vals]:
            return True
    return False


def _run_six_label_task(
    client: OpenAIChatClient,
    sentence: str,
    labels: List[str],
) -> Dict[str, List[Any]]:
    obj, last_raw = parse_with_retries(
        raw_call=lambda: client.chat(SIX_LABEL_SYSTEM_PROMPT, build_six_user_prompt(sentence, labels)),
        parser=_parse_json_object,
        attempts=2,
    )
    if obj is None:
        raise ValueError(
            "six-label extraction JSON object parse failed; "
            f"raw_output={_preview_raw_for_error(last_raw)!r}"
        )
    out: Dict[str, List[Any]] = {}
    for label in labels:
        val = obj.get(label, [])
        out[label] = val if isinstance(val, list) else []
    return out


def _run_single_label_task(
    client: OpenAIChatClient,
    label: str,
    sentence: str,
) -> List[Any]:
    arr, last_raw = parse_with_retries(
        raw_call=lambda: client.chat(SINGLE_LABEL_PROMPTS[label], build_single_user_prompt(label, sentence)),
        parser=_parse_json_array,
        attempts=2,
    )
    if arr is None:
        raise ValueError(
            f"{label} JSON array parse failed; raw_output={_preview_raw_for_error(last_raw)!r}"
        )
    return arr


def run_extract_stage(
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
        raise ValueError("extract-concurrency and extract-files-per-batch must both be >= 1")

    annotate_root = output_dir / target_dir_name / "annotate"
    extract_root = output_dir / target_dir_name / "extract"
    extract_root.mkdir(parents=True, exist_ok=True)

    labeled_root, files = _collect_labeled_jsonl(annotate_root)
    logger.bump("files_total", len(files))
    logger.event(
        "batch_start",
        {
            "annotate_root": str(annotate_root),
            "labeled_root": str(labeled_root),
            "extract_root": str(extract_root),
            "files_total": len(files),
            "extract_concurrency": concurrency,
            "extract_files_per_batch": files_per_batch,
            "extract_mode": mode,
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
        max_tokens=config.EXTRACT_MAX_TOKENS,
    )
    files_processed = 0
    records_written = 0

    for file_batch in chunked(files, files_per_batch):
        logger.event("batch_start", {"batch_files": len(file_batch)})
        for in_file in file_batch:
            rel_path = in_file.relative_to(labeled_root)
            out_file = extract_root / rel_path
            logger.event("file_start", {"file": str(rel_path)})
            file_written = 0
            file_hits = 0
            file_total = 0
            writer = SortedJsonlWriter(out_file)
            try:
                candidates: List[Dict[str, Any]] = []
                try:
                    loaded = load_jsonl_dict_rows_strict(
                        in_file, display_path=str(rel_path)
                    )
                except JsonlValidationError as e:
                    handle_jsonl_validation_and_exit(
                        e,
                        run_error_log,
                        "extract",
                        {"annotate_subpath": str(rel_path)},
                    )
                file_total = len(loaded)
                logger.bump("records_total", file_total)
                schema_issues: List[JsonlLineIssue] = []
                for phys_line, rec in loaded:
                    hit_labels = _target_labels_from_record(rec)
                    has_date = _has_date_only_label(rec)
                    if not hit_labels and not has_date:
                        continue
                    if has_date and not hit_labels:
                        hit_labels = [DATE_ONLY_LABEL]
                    li = _coerce_line_index(rec.get("line_index"))
                    if li is None:
                        schema_issues.append(
                            JsonlLineIssue(
                                str(rel_path),
                                phys_line,
                                "invalid_line_index",
                                "line_index must be int (or whole-number float) when target extraction label is present",
                                "",
                            )
                        )
                        continue
                    file_hits += 1
                    logger.bump("records_hit_target_labels", 1)
                    candidates.append(
                        {
                            "source_file": rec.get("source_file") or str(rel_path),
                            "line_index": li,
                            "sentence": str(rec.get("sentence", "")),
                            "hit_labels": hit_labels,
                            "extraction": {},
                            "has_task_error": False,
                        }
                    )
                if schema_issues:
                    handle_jsonl_validation_and_exit(
                        JsonlValidationError(schema_issues),
                        run_error_log,
                        "extract",
                        {"annotate_subpath": str(rel_path)},
                    )

                resumed_li: Set[int] = set()
                if out_file.exists() and out_file.stat().st_size > 0:
                    try:
                        exist_rows = load_jsonl_records_with_int_line_index_strict(
                            out_file, display_path=str(out_file)
                        )
                    except JsonlValidationError as e:
                        handle_jsonl_validation_and_exit(
                            e,
                            run_error_log,
                            "extract",
                            {
                                "extract_out_file": str(out_file),
                                "context": "resume_load_existing_extract",
                            },
                        )
                    by_li: Dict[int, Dict[str, Any]] = {}
                    for row in exist_rows:
                        by_li[row["line_index"]] = row
                    for one in candidates:
                        li = one["line_index"]
                        if li not in by_li:
                            continue
                        if bool(by_li[li].get("has_task_error", False)):
                            # Failed rows are kept for retry and not counted in resume.
                            continue
                        ex = by_li[li].get("extraction")
                        if isinstance(ex, dict):
                            one["extraction"] = copy.deepcopy(ex)
                        else:
                            one["extraction"] = {}
                        resumed_li.add(li)
                        logger.bump("extract_resume_lines", 1)

                # Task expansion: one request per task; concurrency pool speeds up execution (already-written rows skip API calls)
                six_tasks: List[Tuple[int, str, List[str], str, int | None]] = []
                for idx, one in enumerate(candidates):
                    if one["line_index"] in resumed_li:
                        continue
                    six_hits = [x for x in one["hit_labels"] if x in SIX_TARGET_LABELS]
                    if six_hits:
                        six_tasks.append((idx, one["sentence"], six_hits, one["source_file"], one["line_index"]))

                if six_tasks:
                    with ThreadPoolExecutor(max_workers=concurrency) as executor:
                        futures = {
                            executor.submit(_run_six_label_task, client, sent, labels): (idx, sf, li, labels)
                            for idx, sent, labels, sf, li in six_tasks
                        }
                        for future in tqdm(as_completed(futures), total=len(futures), desc="extract-6", leave=False):
                            idx, sf, li, labels = futures[future]
                            logger.bump("extract_requests_total", 1)
                            try:
                                part = future.result()
                                candidates[idx]["extraction"].update(part)
                            except Exception as err:
                                # Single-task failure should not abort the file; avoids batch exit due to occasional format drift.
                                logger.error(
                                    error_type="extract_six_task_failed",
                                    message=str(err),
                                    source_file=str(sf),
                                    traceback_text=traceback.format_exc(),
                                    line_index=li,
                                    payload={"labels": labels},
                                )
                                append_run_error_log(
                                    run_error_log,
                                    {
                                        "stage": "extract",
                                        "error": str(err),
                                        "source_file": str(sf),
                                        "line_index": li,
                                        "labels": labels,
                                    },
                                )
                                candidates[idx]["has_task_error"] = True
                                for label in labels:
                                    candidates[idx]["extraction"][label] = []

                for label in sorted(SINGLE_LABELS):
                    one_tasks: List[Tuple[int, str, str, int | None]] = []
                    for idx, one in enumerate(candidates):
                        if one["line_index"] in resumed_li:
                            continue
                        if label in one["hit_labels"]:
                            one_tasks.append((idx, one["sentence"], one["source_file"], one["line_index"]))
                    if not one_tasks:
                        continue
                    with ThreadPoolExecutor(max_workers=concurrency) as executor:
                        futures = {
                            executor.submit(_run_single_label_task, client, label, sent): (idx, sf, li)
                            for idx, sent, sf, li in one_tasks
                        }
                        for future in tqdm(as_completed(futures), total=len(futures), desc="extract-1", leave=False):
                            idx, sf, li = futures[future]
                            logger.bump("extract_requests_total", 1)
                            try:
                                arr = future.result()
                                candidates[idx]["extraction"][label] = arr
                            except Exception as err:
                                logger.error(
                                    error_type="extract_single_task_failed",
                                    message=str(err),
                                    source_file=str(sf),
                                    traceback_text=traceback.format_exc(),
                                    line_index=li,
                                    payload={"label": label},
                                )
                                append_run_error_log(
                                    run_error_log,
                                    {
                                        "stage": "extract",
                                        "error": str(err),
                                        "source_file": str(sf),
                                        "line_index": li,
                                        "label": label,
                                    },
                                )
                                candidates[idx]["has_task_error"] = True
                                candidates[idx]["extraction"][label] = []

                # Regex-based date extraction for "Last Updated Date/Effective Date"
                for one in candidates:
                    if one["line_index"] in resumed_li:
                        continue
                    dates = _extract_dates_regex(one["sentence"])
                    if dates:
                        one["extraction"]["Last Updated Date/Effective Date"] = dates

                candidates_written: List[Dict[str, Any]] = []
                for one in candidates:
                    if not one["extraction"]:
                        logger.bump("records_skipped", 1)
                        continue
                    if one["line_index"] in resumed_li:
                        continue
                    out_rec = {
                        "source_file": one["source_file"],
                        "line_index": one["line_index"],
                        "sentence": one["sentence"],
                        "extraction": one["extraction"],
                        "has_task_error": bool(one.get("has_task_error", False)),
                    }
                    file_written += 1
                    records_written += 1
                    logger.bump("records_written", 1)
                    candidates_written.append(out_rec)
                if candidates_written:
                    writer.append(candidates_written)
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
                        "stage": "extract",
                        "error": str(err),
                        "traceback": tb,
                        "source_file": str(rel_path),
                    },
                )
                raise SystemExit(1) from err
            finally:
                writer.flush()
            files_processed += 1
            logger.bump("files_processed", 1)
            logger.event(
                "file_done",
                {
                    "file": str(rel_path),
                    "records_total": file_total,
                    "records_hit_target_labels": file_hits,
                    "records_written": file_written,
                },
            )
            logger.event(
                "progress",
                {
                    "files_processed": files_processed,
                    "files_total": len(files),
                },
            )

    logger.event(
        "run_done",
        {
            "files_processed": files_processed,
            "records_written": records_written,
            "extract_root": str(extract_root),
        },
    )
    return {
        "files_total": len(files),
        "files_processed": files_processed,
        "records_written": records_written,
    }
