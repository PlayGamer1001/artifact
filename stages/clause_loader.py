# -*- coding: utf-8 -*-
"""Load clause taxonomy from clause_labels.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, NamedTuple

from . import config


class Level2Clause(NamedTuple):
    name: str
    definition: str
    example: str = ""


class Level1Category(NamedTuple):
    name: str
    definition: str
    children: List[Level2Clause]
    example: str = ""


def _load_from_json(path: Path) -> List[Level1Category]:
    """Load Level-1 and Level-2 categories from a JSON file."""
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, list) or not data:
        raise ValueError(f"clause labels JSON format error (expected non-empty list): {path}")

    out: List[Level1Category] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        definition = item.get("definition", "")
        example = item.get("example", "")
        children_raw = item.get("children") or []
        if not name:
            continue
        children: List[Level2Clause] = []
        for c in children_raw:
            if isinstance(c, dict) and c.get("name"):
                children.append(
                    Level2Clause(
                        name=str(c["name"]),
                        definition=str(c.get("definition", "")),
                        example=str(c.get("example", "")),
                    )
                )
        out.append(
            Level1Category(
                name=str(name),
                definition=str(definition),
                children=children,
                example=str(example),
            )
        )

    if not out:
        raise ValueError(f"clause labels JSON has no valid Level-1 categories: {path}")
    return out


def load_clauses(
    json_path: Path | None = None,
) -> List[Level1Category]:
    """Load Level-1/Level-2 clause taxonomy from the JSON file."""
    jpath = Path(json_path or config.CLAUSE_JSON_PATH)
    if not jpath.exists():
        raise FileNotFoundError(f"Clause labels file not found: {jpath}")
    return _load_from_json(jpath)

