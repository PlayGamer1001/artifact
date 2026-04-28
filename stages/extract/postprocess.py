# -*- coding: utf-8 -*-
"""Post-processing: normalize Level-1 list and align with Level-2 (keep L1 even if L2 is empty)."""

from __future__ import annotations

from typing import Dict, List, Tuple

from .. import config


def normalize_level1_level2(
    level1_list: List[str],
    level2_dict: Dict[str, List[str]],
) -> Tuple[List[str], Dict[str, List[str]]]:
    """Normalize Level-1/Level-2 results: deduplicate L1, do not drop L1 just because L2 is empty."""
    if not level1_list:
        return [config.NO_LABEL], {config.NO_LABEL: []}

    no_label = config.NO_LABEL
    deduped: List[str] = []
    seen = set()
    for l in level1_list:
        if l in seen:
            continue
        seen.add(l)
        deduped.append(l)

    non_no_l1 = [l for l in deduped if l != no_label]
    if not non_no_l1:
        return [no_label], {no_label: []}

    # If any valid L1 exists, No_label is removed exclusively. Every L1 is kept even if its L2 is empty.
    normalized_l2 = {l: level2_dict.get(l, []) for l in non_no_l1}
    return non_no_l1, normalized_l2
