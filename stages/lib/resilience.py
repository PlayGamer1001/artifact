# -*- coding: utf-8 -*-
"""Resilience utilities: retries for LLM output parsing."""

from __future__ import annotations

import time
from typing import Callable, TypeVar

T = TypeVar("T")


def parse_with_retries(
    raw_call: Callable[[], str],
    parser: Callable[[str], T | None],
    attempts: int = 2,
    backoff_base_sec: float = 0.5,
) -> tuple[T | None, str]:
    """Call raw_call, parse. Retry on failure with backoff."""
    total = max(1, int(attempts))
    last_raw = ""
    for i in range(total):
        raw = raw_call()
        last_raw = raw
        parsed = parser(raw)
        if parsed is not None:
            return parsed, last_raw
        if i < total - 1 and backoff_base_sec > 0:
            time.sleep(backoff_base_sec * (2**i))
    return None, last_raw
