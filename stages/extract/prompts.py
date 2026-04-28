# -*- coding: utf-8 -*-
"""Extract stage system prompt and per-label templates."""

from __future__ import annotations

import json
from typing import Dict, Sequence

SIX_LABEL_SYSTEM_PROMPT = """Task:
Given a privacy-policy sentence or excerpt and a set of assigned labels, extract structured information for the supported labels among the six predefined target labels.

Input:
- `sentence`: a privacy-policy sentence or excerpt, which may contain one or more sentences
- `labels`: assigned labels

Only process the following six target labels:
- Scope of PI Collection: who and whether personal information is collected and what categories are involved
- Sources of PI Collected: where collected personal information comes from
- Scope of PI Sold/Shared/Disclosed: who and whether personal information is sold, shared, or disclosed and what categories are involved
- Third-Party Recipients of PI: which third parties receive personal information
- Purposes for PI Processing: why personal information is collected, used, or processed
- PI Retention Periods/Criteria: how long personal information is retained, or what criteria determine retention

Ignore any input labels outside these six target labels.

Output format:
Always return a JSON object whose keys are exactly the labels provided in the input.
For each label:
- if the sentence supports that label, return a list of extracted object(s)
- if the sentence does not support that label, return an empty list `[]` as the value for that key

Example when no label is supported:
{"Scope of PI Collection": [], "Sources of PI Collected": [], ...}

Never return a bare [] or {} — the response must always be a JSON object with the input label keys.
Do not output labels that are not in the input.

General rules:
- Extract only information explicitly stated in the sentence.
- Keep extracted strings as close as possible to the original wording.
- Omit any non-boolean field that is not explicitly supported.
- For boolean fields, output `true` only when explicitly supported; otherwise output `false`.

Field definitions:
- `action`: verbs describing what is done to personal information, such as collect, use, disclose, share, sell, or retain
- `action_negation`: whether the action is explicitly negated, such as "do not collect" or "never sell"
- `data`: categories or types of personal information mentioned in the sentence
- `source`: who or where the personal information is collected from
- `purpose`: reasons for collecting, using, or processing personal information
- `recipient`: third parties that receive personal information
- `retention`: explicit retention periods, time expressions, or criteria describing how long personal information is retained
- `context`: explicit conditional clauses that state when the processing applies, such as "if you register an account" or "when you use our services". Do not treat source phrases, subject groups, channels, methods, purposes, recipients, or other prepositional modifiers as `context`.
- `actor`: who collects the personal information, such as the first-party company itself, a subsidiary, or a third party

Do not treat source phrases, channels, methods, purposes, or recipient descriptions as `context`.

Field schema by label:

- Scope of PI Collection
- Scope of PI Sold/Shared/Disclosed
  - `action`: List[str]
  - `action_negation`: bool
  - `actor`: List[str]
  - `data`: List[str]
  - `context`: List[str]

- Sources of PI Collected
  - `source`: List[str]
  - `data`: List[str]

- Third-Party Recipients of PI
  - `action`: List[str]
  - `action_negation`: bool
  - `data`: List[str]
  - `recipient`: List[str]

- Purposes for PI Processing
  - `data`: List[str]
  - `purpose`: List[str]

- PI Retention Periods/Criteria
  - `data`: List[str]
  - `retention`: List[str]

Return JSON only.
"""


def build_six_user_prompt(sentence: str, labels: Sequence[str]) -> str:
    labels_text = json.dumps(list(labels), ensure_ascii=False)
    return f"Input:\n- sentence: {sentence}\n- labels: {labels_text}\n\nReturn JSON only."