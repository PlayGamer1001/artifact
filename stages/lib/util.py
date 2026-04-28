# -*- coding: utf-8 -*-
"""Utility functions."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, TypeVar

T = TypeVar("T")


def chunked(seq: List[T], size: int) -> Iterator[List[T]]:
    """Split a list into fixed-size chunks; the last chunk may be shorter."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def should_skip_jsonl_path(path: Path) -> bool:
    """Return True if this jsonl path should be skipped (hidden dirs, checkpoints, cache, etc.)."""
    parts = path.parts
    for p in parts:
        if p.startswith("."):
            return True
        if p == "__pycache__":
            return True
    return False
