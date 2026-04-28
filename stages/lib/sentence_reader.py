# -*- coding: utf-8 -*-
"""
Iterate all .jsonl files under PP_jsonl, yielding (source_file, line_index, sentence, heading_context).

Input jsonl: one sentence object per line. Key fields:
- sentence: sentence text
- section: section heading of the sentence (optional)
"""

import json
from pathlib import Path
from typing import Iterator, List, Set, Tuple

from .. import config
from .jsonl_strict import JsonlValidationError, collect_pp_jsonl_structural_issues
from .util import should_skip_jsonl_path


def _resolve_target_files(
    root: Path,
    include_source_files: Set[str] | None,
) -> List[tuple[str, Path]]:
    """Return list of (source_file, absolute_path) to be read."""
    if include_source_files is None:
        out: List[tuple[str, Path]] = []
        for jsonl_path in sorted(root.rglob("*.jsonl")):
            if should_skip_jsonl_path(jsonl_path):
                continue
            try:
                source_file = jsonl_path.relative_to(root).as_posix()
            except ValueError:
                source_file = jsonl_path.name
            out.append((source_file, jsonl_path))
        return out

    out: List[tuple[str, Path]] = []
    root_resolved = root.resolve()
    for source_file in sorted(include_source_files):
        src = source_file.replace("\\", "/").lstrip("/")
        abs_path = (root / src).resolve()
        try:
            abs_path.relative_to(root_resolved)
        except Exception:
            # Illegal path traversal detected; skip.
            continue
        if abs_path.exists() and abs_path.is_file():
            out.append((src, abs_path))
    return out


def iter_sentences(
    pp_jsonl_root: Path | None = None,
    skip_empty: bool | None = None,
    min_length: int | None = None,
    include_source_files: Set[str] | None = None,
) -> Iterator[Tuple[str, int, str, List[str]]]:
    """
    Iterate all .jsonl files under the PP_jsonl root, yielding (source_file, line_index, sentence, heading_context).
    - sentence is read from the "sentence" field of each JSON line.
    - heading_context is generated from the "section" field: [section] if present, else empty list.
    """
    root = Path(pp_jsonl_root or config.PP_JSONL_ROOT)
    if skip_empty is None:
        skip_empty = config.SKIP_EMPTY_LINES
    if min_length is None:
        min_length = config.MIN_SENTENCE_LENGTH

    targets = _resolve_target_files(root, include_source_files)
    for source_file, jsonl_path in targets:
        issues = collect_pp_jsonl_structural_issues(jsonl_path, display_path=source_file)
        if issues:
            raise JsonlValidationError(issues)

        with open(jsonl_path, "r", encoding="utf-8", errors="replace") as fp2:
            for idx, line in enumerate(fp2):
                raw = line.strip()
                if not raw:
                    if skip_empty:
                        continue
                    if min_length > 0:
                        continue
                    yield source_file, idx, "", []
                    continue

                item = json.loads(raw)
                sentence = str(item.get("sentence", "")).strip()
                if not sentence:
                    if skip_empty:
                        continue
                    if min_length > 0:
                        continue

                if min_length > 0 and len(sentence) < min_length:
                    continue

                section = item.get("section")
                if isinstance(section, str) and section.strip():
                    heading_context = [section.strip()]
                else:
                    heading_context = []

                yield source_file, idx, sentence, heading_context
