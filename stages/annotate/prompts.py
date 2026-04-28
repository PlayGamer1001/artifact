# -*- coding: utf-8 -*-
"""Annotation stage (L1/L2) system and user prompt templates."""

from __future__ import annotations

from typing import List

from ..clause_loader import Level1Category, Level2Clause

COMMON_CLASSIFICATION_RULES = """- Use the sentence itself as the primary basis.
- Use heading context only to resolve ambiguity in the sentence.
- Do not assign a label if the sentence itself does not express the substance of that label.
- A heading may help interpret shorthand references, but a heading alone is never sufficient evidence.
- Do not infer unstated meaning.
- Do not assign labels to section titles, generic introductions, generic cross-references, or boilerplate statements that do not state a concrete data practice, right, mechanism, or policy fact.
"""


LEVEL1_SYSTEM_TEMPLATE = """You are an expert classifier for privacy policy text.

Your task is to assign zero or more level-1 categories to a single sentence from a privacy policy.

You will receive:
- one sentence from a privacy policy
- optional heading context
- the list of allowed level-1 categories with their definitions

{categories_list}

Instructions:
- Classify the sentence according to the explicit information it contains.
{common_rules}
- Assign multiple labels when the sentence explicitly expresses more than one applicable category, even if they appear in the same sentence.
- If no category applies, return ["No_label"].

Output requirements:
- Return valid JSON only.
- Output must be a JSON array of level-1 category names using exactly the names provided.
- Do not output explanations or extra text.
"""

LEVEL1_USER_TEMPLATE = """Classify the following sentence into the applicable level-1 categories.
{context_block}
Sentence:
{sentence}

Return JSON only.
"""

LEVEL1_USER_CONTEXT_BLOCK = """heading_context:
{heading_context}
"""

LEVEL2_MULTI_SYSTEM_TEMPLATE = """You are an expert classifier for privacy policy text.

Your task is to assign all applicable level-2 clauses for a sentence, but only within the provided level-1 category.

Instructions:
{common_rules}
- Assign multiple labels when the sentence explicitly expresses more than one applicable category or clause, even if they appear in the same sentence.
- If none of the candidate clauses apply, return [].

Output requirements:
- Return valid JSON only.
- Output must be a JSON array of level-2 clause names using exactly the candidate names provided.
- Do not output explanations or extra text.
"""

LEVEL2_MULTI_USER_TEMPLATE = """Classify the following sentence into applicable level-2 clauses under the provided level-1 category.
{context_block}
Sentence:
{sentence}

Level-1 category:
{level1_name}

Candidate level-2 clauses:
{candidate_definitions}

Return JSON only.
"""

LEVEL2_USER_CONTEXT_BLOCK = """heading_context:
{heading_context}
"""


def build_level1_system(categories: List[Level1Category]) -> str:
    lines = []
    for c in categories:
        lines.append(f"- {c.name}: {c.definition}")
    return LEVEL1_SYSTEM_TEMPLATE.format(
        categories_list="\n".join(lines),
        common_rules=COMMON_CLASSIFICATION_RULES.strip(),
    )


def build_level1_user(sentence: str, heading_context: List[str] | None = None) -> str:
    text = sentence.strip()
    if heading_context:
        context_block = LEVEL1_USER_CONTEXT_BLOCK.format(
            heading_context=" / ".join(heading_context)
        )
    else:
        context_block = ""
    return LEVEL1_USER_TEMPLATE.format(context_block=context_block, sentence=text)


def _format_l2_candidates(level2_options: List[Level2Clause]) -> str:
    lines = []
    for clause in level2_options:
        lines.append(
            f"- {clause.name}\n"
            f"  Definition: {clause.definition}\n"
            f"  Example: {clause.example}"
        )
    return "\n".join(lines)


def build_level2_multi_system() -> str:
    return LEVEL2_MULTI_SYSTEM_TEMPLATE.format(
        common_rules=COMMON_CLASSIFICATION_RULES.strip(),
    )


def build_level2_multi_user(
    sentence: str,
    level1_name: str,
    level2_options: List[Level2Clause],
    heading_context: List[str] | None = None,
) -> str:
    text = sentence.strip()
    if heading_context:
        context_block = LEVEL2_USER_CONTEXT_BLOCK.format(
            heading_context=" / ".join(heading_context)
        )
    else:
        context_block = ""
    return LEVEL2_MULTI_USER_TEMPLATE.format(
        candidate_definitions=_format_l2_candidates(level2_options),
        level1_name=level1_name,
        context_block=context_block,
        sentence=text,
    )
