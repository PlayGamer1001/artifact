# -*- coding: utf-8 -*-
"""Normalize stage: valid label sets and output validation (prefix unmatched LLM returns with 'Other:', no programmatic fallback)."""

from __future__ import annotations

# Label names must match the leaf node names in each field's label block from normalize_prompts (used for whitelist validation)
DATA_LABELS: frozenset[str] = frozenset(
    {
        "Data(Unspecified)",
        "De-Identified Data",
        "Publicly Available Information",
        "Aggregated Data",
        "Personal Identifier",
        "Commercial Information",
        "Internet/Electronic Network Activity",
        "Sensory Data",
        "Consumer Profile",
        "Coarse/Approximate Location",
        "Professional/Employment-Related Information",
        "Sensitive Data",
        "Biometric Data",
    }
)

ENTITY_LABELS: frozenset[str] = frozenset(
    {
        "First-Party",
        "Consumer",
        "Affiliates",
        "Third-Party(Unspecified)",
        "Advertising Networks",
        "Analytics Providers",
        "Authentication Providers",
        "Content Providers",
        "Email Service Providers",
        "Government Entities",
        "Internet Service Providers",
        "Operating Systems/Platforms",
        "SDK Providers",
        "Social Networks",
        "Payment Processors",
        "Data Brokers",
    }
)

PURPOSE_LABELS: frozenset[str] = frozenset(
    {
        "Services",
        "Security",
        "Legal",
        "Advertising/Marketing",
        "Analytics/Research",
        "Personalization/Customization",
        "Merger/Acquisition",
    }
)


def field_kind(field: str) -> str:
    if field == "data":
        return "data"
    if field in ("source", "recipient", "actor"):
        return "entity"
    if field == "purpose":
        return "purpose"
    raise ValueError(f"unsupported field: {field}")


def normalize_label(field: str, raw: str) -> tuple[str, bool]:
    """Returns (output_label, is_in_whitelist).

    ``raw`` must be the normalize stage LLM (API) raw return string, **not** the ``extracted_text`` from the extract slot.
    On whitelist hit (case-insensitive), returns canonical form and True; otherwise if ``raw`` is non-empty after strip,
    returns ``Other: `` + the stripped **LLM output** and False; if LLM returns empty (empty after strip), returns "" and False.
    """
    kind = field_kind(field)
    s = raw.strip().strip('"').strip("'").rstrip(".")
    if kind == "data":
        allowed = DATA_LABELS
    elif kind == "entity":
        allowed = ENTITY_LABELS
    else:
        allowed = PURPOSE_LABELS

    if s in allowed:
        return s, True
    for a in allowed:
        if a.lower() == s.lower():
            return a, True
    if s:
        # s is from raw (normalize API return), not from extract's extracted_text
        return f"Other: {s}", False
    return "", False
