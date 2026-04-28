# -*- coding: utf-8 -*-
"""Compute sentence token distribution with tiktoken."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import tiktoken


def iter_jsonl_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for p in sorted(path.rglob("*.jsonl")):
        if p.is_file():
            yield p


def load_encoder(model: str, encoding: str | None):
    if encoding:
        return tiktoken.get_encoding(encoding)
    return tiktoken.encoding_for_model(model)


def bucket_name(n: int) -> str:
    if n == 0:
        return "0"
    if n <= 16:
        return "1-16"
    if n <= 32:
        return "17-32"
    if n <= 64:
        return "33-64"
    if n <= 128:
        return "65-128"
    if n <= 256:
        return "129-256"
    if n <= 512:
        return "257-512"
    if n <= 1024:
        return "513-1024"
    if n <= 2048:
        return "1025-2048"
    return "2049+"


def percentile(sorted_vals: List[int], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    idx = q * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def run_token_distribution(
    input_path: Path,
    output_path: Path,
    gt512_output: Path,
    model: str = "gpt-4o-mini",
    encoding: str | None = "cl100k_base",
) -> dict:
    """Compute token distribution statistics for JSONL files."""
    if not input_path.exists():
        raise FileNotFoundError(f"input path not found: {input_path}")

    enc = load_encoder(model, encoding)
    counts: List[int] = []
    bucket_counts: Dict[str, int] = {
        "0": 0,
        "1-16": 0,
        "17-32": 0,
        "33-64": 0,
        "65-128": 0,
        "129-256": 0,
        "257-512": 0,
        "513-1024": 0,
        "1025-2048": 0,
        "2049+": 0,
    }
    files_scanned = 0
    records_scanned = 0
    invalid_records = 0
    gt512_source_type_counts: Dict[str, int] = {}

    gt512_output.parent.mkdir(parents=True, exist_ok=True)
    gt512_out = gt512_output.open("w", encoding="utf-8", newline="\n")

    try:
        for fp in iter_jsonl_files(input_path):
            files_scanned += 1
            docid = fp.stem
            with fp.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    records_scanned += 1
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        invalid_records += 1
                        continue
                    if not isinstance(obj, dict):
                        invalid_records += 1
                        continue
                    sentence = obj.get("sentence", "")
                    if not isinstance(sentence, str):
                        sentence = str(sentence)
                    source_type = obj.get("source_type", "unknown")
                    if not isinstance(source_type, str):
                        source_type = str(source_type)
                    sentence_id = obj.get("sentence_id", "")
                    if not isinstance(sentence_id, str):
                        sentence_id = str(sentence_id)
                    n = len(enc.encode(sentence))
                    counts.append(n)
                    bucket_counts[bucket_name(n)] += 1
                    if n > 512:
                        gt512_source_type_counts[source_type] = gt512_source_type_counts.get(source_type, 0) + 1
                        rec = {
                            "docid": docid,
                            "sentence_id": sentence_id,
                            "source_type": source_type,
                            "token_count": n,
                            "sentence": sentence,
                        }
                        gt512_out.write(json.dumps(rec, ensure_ascii=False))
                        gt512_out.write("\n")
    finally:
        gt512_out.close()

    counts_sorted = sorted(counts)
    total = len(counts_sorted)

    summary = {
        "input": str(input_path),
        "files_scanned": files_scanned,
        "records_scanned": records_scanned,
        "valid_sentences": total,
        "invalid_records": invalid_records,
        "model": model,
        "encoding": encoding or enc.name,
        "basic_stats": {
            "min": counts_sorted[0] if total else 0,
            "max": counts_sorted[-1] if total else 0,
            "mean": (sum(counts_sorted) / total) if total else 0.0,
            "p50": percentile(counts_sorted, 0.50),
            "p90": percentile(counts_sorted, 0.90),
            "p95": percentile(counts_sorted, 0.95),
            "p99": percentile(counts_sorted, 0.99),
        },
        "threshold_counts": {
            "eq_0": sum(1 for x in counts_sorted if x == 0),
            "gt_512": sum(1 for x in counts_sorted if x > 512),
            "gt_1024": sum(1 for x in counts_sorted if x > 1024),
            "gt_2048": sum(1 for x in counts_sorted if x > 2048),
        },
        "gt_512_source_type_counts": gt512_source_type_counts,
        "bucket_counts": bucket_counts,
        "gt_512_records_jsonl": str(gt512_output),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    return summary
