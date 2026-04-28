# -*- coding: utf-8 -*-
"""Write annotation results to JSONL."""

import json
from pathlib import Path
from typing import Any, Dict, List

from .jsonl_strict import (
    load_jsonl_dict_rows_strict,
    load_jsonl_records_with_int_line_index_strict,
)


def write_jsonl_to_file(
    records: List[Dict[str, Any]],
    output_file: Path,
) -> Path:
    """Write multiple records to a JSONL file at once. Creates parent directories. Returns the written file path."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return output_file


def append_jsonl_to_file(
    records: List[Dict[str, Any]],
    output_file: Path,
) -> Path:
    """Append multiple records to a JSONL file. Creates parent directories. Returns the file path."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return output_file


def normalize_jsonl_file_by_line_index(output_file: Path) -> Path:
    """Deduplicate by line_index: keep only the last occurrence of each line_index, then sort by line_index and write back."""
    output_file = Path(output_file)
    if not output_file.exists():
        return output_file

    disp = str(output_file)
    records = load_jsonl_records_with_int_line_index_strict(output_file, display_path=disp)
    indexed_latest: Dict[int, Dict[str, Any]] = {}
    for rec in records:
        indexed_latest[rec["line_index"]] = rec
    ordered_records: List[Dict[str, Any]] = [indexed_latest[li] for li in sorted(indexed_latest)]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = output_file.parent / f"{output_file.name}.tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        for rec in ordered_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp_file.replace(output_file)
    return output_file


def normalize_pairs_jsonl_file_by_slot(output_file: Path) -> Path:
    """Deduplicate by normalize slot: keep only the last occurrence of each slot, then sort by slot and write back."""
    output_file = Path(output_file)
    if not output_file.exists():
        return output_file

    disp = str(output_file)
    rows = load_jsonl_dict_rows_strict(output_file, display_path=disp)
    indexed_latest: Dict[tuple[str, str, str, str, str, str], Dict[str, Any]] = {}
    for _line_no, rec in rows:
        key = (
            str(rec.get("source_file", "")),
            str(rec.get("line_index", "")),
            str(rec.get("ppi_key", "")),
            str(rec.get("object_index", "")),
            str(rec.get("field", "")),
            str(rec.get("element_index", "")),
        )
        indexed_latest[key] = rec
    ordered_records: List[Dict[str, Any]] = [
        indexed_latest[k] for k in sorted(indexed_latest)
    ]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = output_file.parent / f"{output_file.name}.tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        for rec in ordered_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp_file.replace(output_file)
    return output_file


class SortedJsonlWriter:
    """Accumulate records in memory (deduped by line_index), write sorted on flush.

    Usage:
        writer = SortedJsonlWriter(out_file)
        writer.append(new_records)   # dedup by line_index automatically
        writer.flush()               # write sorted once, clear buffer
    """

    def __init__(self, output_file: Path):
        self.output_file = Path(output_file)
        self._buf: Dict[int, Dict[str, Any]] = {}
        self._out_file_existed = (
            self.output_file.exists() and self.output_file.stat().st_size > 0
        )
        if self._out_file_existed:
            for rec in load_jsonl_records_with_int_line_index_strict(
                self.output_file, display_path=str(self.output_file)
            ):
                self._buf[rec["line_index"]] = rec

    def append(self, records: List[Dict[str, Any]]) -> None:
        for rec in records:
            self._buf[rec["line_index"]] = rec

    def flush(self) -> None:
        if not self._buf:
            return
        ordered = [self._buf[li] for li in sorted(self._buf)]
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = self.output_file.parent / f"{self.output_file.name}.tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            for rec in ordered:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp_file.replace(self.output_file)
        self._buf.clear()


class SortedPairsJsonlWriter:
    """Accumulate normalize-pair records in memory (deduped by slot), write sorted on flush.

    Usage:
        writer = SortedPairsJsonlWriter(out_file)
        writer.append(new_records)   # dedup by (source_file, line_index, ppi_key, object_index, field, element_index)
        writer.flush()               # write sorted once, clear buffer
    """

    def __init__(self, output_file: Path):
        self.output_file = Path(output_file)
        self._buf: Dict[tuple, Dict[str, Any]] = {}
        self._out_file_existed = (
            self.output_file.exists() and self.output_file.stat().st_size > 0
        )
        if self._out_file_existed:
            for _line_no, rec in load_jsonl_dict_rows_strict(
                self.output_file, display_path=str(self.output_file)
            ):
                key = (
                    str(rec.get("source_file", "")),
                    str(rec.get("line_index", "")),
                    str(rec.get("ppi_key", "")),
                    str(rec.get("object_index", "")),
                    str(rec.get("field", "")),
                    str(rec.get("element_index", "")),
                )
                self._buf[key] = rec

    def append(self, records: List[Dict[str, Any]]) -> None:
        for rec in records:
            key = (
                str(rec.get("source_file", "")),
                str(rec.get("line_index", "")),
                str(rec.get("ppi_key", "")),
                str(rec.get("object_index", "")),
                str(rec.get("field", "")),
                str(rec.get("element_index", "")),
            )
            self._buf[key] = rec

    def flush(self) -> None:
        if not self._buf:
            return
        ordered = [self._buf[k] for k in sorted(self._buf)]
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = self.output_file.parent / f"{self.output_file.name}.tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            for rec in ordered:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp_file.replace(self.output_file)
        self._buf.clear()
