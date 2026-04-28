# -*- coding: utf-8 -*-
"""Strict JSONL reader: non-empty lines must be valid JSON objects; collects all malformed lines before erroring."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Set

from .runtime_logger import append_run_error_log


@dataclass(frozen=True)
class JsonlLineIssue:
    path: str
    line_no: int
    kind: str
    message: str
    preview: str = ""


class JsonlValidationError(Exception):
    """File-level validation failure; carries all problem lines in the file."""

    def __init__(self, issues: List[JsonlLineIssue]):
        self.issues = issues
        super().__init__(f"JSONL validation failed, {len(issues)} issue(s) found")


class JsonlValidationFatal(SystemExit):
    """Unrecoverable JSONL validation error — triggers finally blocks before exit.

    Subclass of SystemExit so it propagates through except Exception blocks
    but is caught by bare `except SystemExit` or `except BaseException`.
    """

    def __init__(self, caused_by: Exception):
        self.caused_by = caused_by
        super().__init__(1)



def _iter_nonempty_stripped_lines(path: Path) -> Iterator[tuple[int, str]]:
    """(1-based physical line number, non-empty content after strip)."""
    with open(path, "r", encoding="utf-8", errors="replace") as fp:
        for line_no, raw in enumerate(fp, start=1):
            s = raw.strip()
            if not s:
                continue
            yield line_no, s


def collect_pp_jsonl_structural_issues(path: Path, *, display_path: str) -> List[JsonlLineIssue]:
    """PP input jsonl: each non-empty line must be valid JSON with a dict at the top level."""
    issues: List[JsonlLineIssue] = []
    for line_no, s in _iter_nonempty_stripped_lines(path):
        try:
            obj = json.loads(s)
        except json.JSONDecodeError as e:
            issues.append(
                JsonlLineIssue(
                    display_path,
                    line_no,
                    "json_decode_error",
                    str(e),
                    s[:240],
                )
            )
            continue
        if not isinstance(obj, dict):
            issues.append(
                JsonlLineIssue(
                    display_path,
                    line_no,
                    "not_json_object",
                    f"top-level type is {type(obj).__name__}, must be object",
                    s[:240],
                )
            )
    return issues


def load_jsonl_dict_rows_strict(
    path: Path, *, display_path: str
) -> List[tuple[int, Dict[str, Any]]]:
    """Each non-empty line must be a JSON object; collects all file issues before raising. Returns (line_no, object) list."""
    issues: List[JsonlLineIssue] = []
    rows: List[tuple[int, Dict[str, Any]]] = []
    for line_no, s in _iter_nonempty_stripped_lines(path):
        try:
            obj = json.loads(s)
        except json.JSONDecodeError as e:
            issues.append(JsonlLineIssue(display_path, line_no, "json_decode_error", str(e), s[:240]))
            continue
        if not isinstance(obj, dict):
            issues.append(
                JsonlLineIssue(
                    display_path,
                    line_no,
                    "not_json_object",
                    f"top-level type is {type(obj).__name__}",
                    s[:240],
                )
            )
            continue
        rows.append((line_no, obj))
    if issues:
        raise JsonlValidationError(issues)
    return rows


def _line_index_valid(v: Any) -> bool:
    return type(v) is int


def load_jsonl_records_with_int_line_index_strict(
    path: Path, *, display_path: str
) -> List[Dict[str, Any]]:
    """Each non-empty line must be a dict with an int line_index (bool excluded)."""
    issues: List[JsonlLineIssue] = []
    rows: List[Dict[str, Any]] = []
    for line_no, s in _iter_nonempty_stripped_lines(path):
        try:
            obj = json.loads(s)
        except json.JSONDecodeError as e:
            issues.append(JsonlLineIssue(display_path, line_no, "json_decode_error", str(e), s[:240]))
            continue
        if not isinstance(obj, dict):
            issues.append(
                JsonlLineIssue(
                    display_path,
                    line_no,
                    "not_json_object",
                    f"top-level type is {type(obj).__name__}",
                    s[:240],
                )
            )
            continue
        li = obj.get("line_index")
        if not _line_index_valid(li):
            issues.append(
                JsonlLineIssue(
                    display_path,
                    line_no,
                    "invalid_line_index",
                    f"line_index must be int, got {type(li).__name__}: {li!r}",
                    s[:240],
                )
            )
            continue
        rows.append(obj)
    if issues:
        raise JsonlValidationError(issues)
    return rows


def load_stage1_l1_records_strict(
    path: Path, *, source_file: str, display_path: str
) -> List[Dict[str, Any]]:
    """stage1 intermediate file: each line is a dict with line_index(int), sentence(str), level1(list)."""
    issues: List[JsonlLineIssue] = []
    out: List[Dict[str, Any]] = []
    for line_no, s in _iter_nonempty_stripped_lines(path):
        try:
            obj = json.loads(s)
        except json.JSONDecodeError as e:
            issues.append(JsonlLineIssue(display_path, line_no, "json_decode_error", str(e), s[:240]))
            continue
        if not isinstance(obj, dict):
            issues.append(
                JsonlLineIssue(
                    display_path,
                    line_no,
                    "not_json_object",
                    f"top-level type is {type(obj).__name__}",
                    s[:240],
                )
            )
            continue
        level1 = obj.get("level1")
        sentence = obj.get("sentence")
        line_index = obj.get("line_index")
        if not isinstance(level1, list):
            issues.append(
                JsonlLineIssue(
                    display_path,
                    line_no,
                    "invalid_stage1_schema",
                    "level1 must be list",
                    s[:240],
                )
            )
            continue
        if not isinstance(sentence, str):
            issues.append(
                JsonlLineIssue(
                    display_path,
                    line_no,
                    "invalid_stage1_schema",
                    "sentence must be str",
                    s[:240],
                )
            )
            continue
        if not _line_index_valid(line_index):
            issues.append(
                JsonlLineIssue(
                    display_path,
                    line_no,
                    "invalid_stage1_schema",
                    f"line_index must be int, got {type(line_index).__name__}",
                    s[:240],
                )
            )
            continue
        heading_context = obj.get("heading_context")
        if heading_context is not None and not isinstance(heading_context, list):
            issues.append(
                JsonlLineIssue(
                    display_path,
                    line_no,
                    "invalid_stage1_schema",
                    "heading_context must be list if present",
                    s[:240],
                )
            )
            continue
        if heading_context is None:
            heading_context = []
        level1_strs = [str(x) for x in level1]
        has_task_error = bool(obj.get("has_task_error", False))
        out.append(
            {
                "source_file": source_file,
                "line_index": line_index,
                "sentence": sentence,
                "heading_context": list(heading_context),
                "level1": level1_strs,
                "has_task_error": has_task_error,
            }
        )
    if issues:
        raise JsonlValidationError(issues)
    return out


def load_l2_done_line_indices_strict(path: Path, *, display_path: str) -> Set[int]:
    """`_state/l2_done_items/*.jsonl`: each non-empty line must be a canonical decimal integer string."""
    if not path.exists():
        return set()
    issues: List[JsonlLineIssue] = []
    out: Set[int] = set()
    with open(path, "r", encoding="utf-8", errors="replace") as fp:
        for line_no, raw in enumerate(fp, start=1):
            s = raw.strip()
            if not s:
                continue
            if not re.fullmatch(r"-?\d+", s):
                issues.append(
                    JsonlLineIssue(
                        display_path,
                        line_no,
                        "invalid_l2_done_line",
                        "non-empty line must be integer text (optional sign + digits)",
                        s[:80],
                    )
                )
                continue
            v = int(s)
            if str(v) != s:
                issues.append(
                    JsonlLineIssue(
                        display_path,
                        line_no,
                        "invalid_l2_done_line",
                        f"non-canonical integer literal: {s!r}",
                        s[:80],
                    )
                )
                continue
            out.add(v)
    if issues:
        raise JsonlValidationError(issues)
    return out


def load_done_line_index_set_from_answer_jsonl(path: Path, *, display_path: str) -> Set[int]:
    """Return line_indices where has_task_error == False (i.e., successfully processed)."""
    rows = load_jsonl_records_with_int_line_index_strict(path, display_path=display_path)
    return {r["line_index"] for r in rows if not bool(r.get("has_task_error", False))}


def handle_jsonl_validation_and_exit(
    err: JsonlValidationError,
    run_error_log: Path,
    stage: str,
    extra: Dict[str, Any] | None = None,
) -> None:
    payload: Dict[str, Any] = {
        "stage": stage,
        "error_type": "jsonl_validation_failed",
        "issue_count": len(err.issues),
        "issues": [asdict(i) for i in err.issues],
    }
    if extra:
        payload.update(extra)
    append_run_error_log(run_error_log, payload)
    for i in err.issues:
        print(f"[ERROR] {i.path} line {i.line_no} [{i.kind}] {i.message}")
    raise JsonlValidationFatal(err) from err
