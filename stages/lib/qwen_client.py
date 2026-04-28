# -*- coding: utf-8 -*-
"""OpenAI-compatible client for Qwen model, providing classify_level1 and classify_level2 (multi-select within L2 group)."""

import json
import re
from dataclasses import dataclass
from typing import Dict, List

from openai import APIStatusError, OpenAI

from .. import config
from ..clause_loader import Level1Category, Level2Clause
from ..annotate.prompts import (
    build_level1_system,
    build_level1_user,
    build_level2_multi_system,
    build_level2_multi_user,
)
from .resilience import parse_with_retries


@dataclass(frozen=True)
class SamplingConfig:
    """Sampling parameter configuration."""

    temperature: float
    top_p: float
    top_k: int
    min_p: float
    presence_penalty: float
    repetition_penalty: float
    enable_thinking: bool


def _mode_to_enable_thinking(mode: str) -> bool:
    """Convert mode string to enable_thinking boolean."""
    m = str(mode).strip().lower()
    if m == "thinking":
        return True
    if m == "nothinking":
        return False
    raise ValueError(f"unsupported mode: {mode}")


def sampling_from_mode(mode: str) -> SamplingConfig:
    """Build SamplingConfig from 'thinking' or 'nothinking' mode, matching config values."""
    enable_thinking = _mode_to_enable_thinking(mode)
    if enable_thinking:
        return SamplingConfig(
            temperature=config.THINKING_TEMPERATURE,
            top_p=config.THINKING_TOP_P,
            top_k=config.THINKING_TOP_K,
            min_p=config.THINKING_MIN_P,
            presence_penalty=config.THINKING_PRESENCE_PENALTY,
            repetition_penalty=config.THINKING_REPETITION_PENALTY,
            enable_thinking=config.THINKING_ENABLE_THINKING,
        )
    return SamplingConfig(
        temperature=config.NOTHINKING_TEMPERATURE,
        top_p=config.NOTHINKING_TOP_P,
        top_k=config.NOTHINKING_TOP_K,
        min_p=config.NOTHINKING_MIN_P,
        presence_penalty=config.NOTHINKING_PRESENCE_PENALTY,
        repetition_penalty=config.NOTHINKING_REPETITION_PENALTY,
        enable_thinking=config.NOTHINKING_ENABLE_THINKING,
    )


class OpenAIChatClient:
    """OpenAI-compatible Chat Completions: single round system + user prompt. Reused by extract/normalize."""

    def __init__(
        self,
        base_url: str,
        model: str,
        sampling: SamplingConfig,
        timeout: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self._client = OpenAI(base_url=base_url, api_key=config.QWEN_API_KEY)
        self._model = model
        self._timeout = timeout if timeout is not None else config.QWEN_TIMEOUT
        self._sampling = sampling
        self._max_tokens = max_tokens if max_tokens is not None else config.EXTRACT_MAX_TOKENS

    @staticmethod
    def _raise_on_vllm_fatal(err: Exception) -> None:
        msg = str(err)
        if isinstance(err, APIStatusError) and err.status_code is not None and err.status_code >= 500:
            raise RuntimeError(
                f"vLLM server returned {err.status_code}; fail-fast exit."
            ) from err
        if "EngineDeadError" in msg or "EngineCore encountered an issue" in msg:
            raise RuntimeError("vLLM EngineDeadError detected; fail-fast exit.") from err
        raise err

    def chat(self, system: str, user: str) -> str:
        return _chat_once(
            client=self._client,
            model=self._model,
            timeout=self._timeout,
            sampling=self._sampling,
            max_tokens=self._max_tokens,
            system=system,
            user=user,
        )


def _parse_json_array(text: str) -> List[str]:
    """Parse JSON array from model output. Tolerates whitespace, markdown code fences, etc."""
    text = (text or "").strip()
    # Strip possible markdown code fences
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()
    try:
        arr = json.loads(text)
        if isinstance(arr, list):
            return [str(x).strip() for x in arr if str(x).strip()]
        return []
    except json.JSONDecodeError:
        # Try to match [...] content
        m = re.search(r"\[[\s\S]*\]", text)
        if m:
            try:
                arr = json.loads(m.group(0))
                if isinstance(arr, list):
                    return [str(x).strip() for x in arr if str(x).strip()]
            except json.JSONDecodeError:
                pass
        return []


def _parse_level1_result(
    text: str,
    categories: List[Level1Category],
) -> tuple[List[str], bool]:
    """Parse L1 output. Returns (labels, parsed_ok). parsed_ok=False means format error."""
    labels = _parse_json_array(text)
    # Even an explicit [] parse is considered parsed_ok (will normalize to No_label later)
    if not labels and "[]" not in (text or ""):
        return ([], False)
    valid = {c.name for c in categories} | {config.NO_LABEL}
    # If model outputs an array but ALL labels are invalid, treat as parse failure to avoid silently polluting results with No_label.
    if labels and not any(l in valid for l in labels):
        return ([], False)
    return (_normalize_level1_labels(labels, valid), True)


def _normalize_level1_labels(labels: List[str], valid: set[str]) -> List[str]:
    """Normalize Level-1 labels: filter invalid, deduplicate, handle No_label exclusivity, fallback to [No_label]."""
    out: List[str] = []
    seen: set[str] = set()
    for l in labels:
        if l not in valid or l in seen:
            continue
        seen.add(l)
        out.append(l)
    no_label = config.NO_LABEL
    non_no = [l for l in out if l != no_label]
    if non_no:
        return non_no
    if no_label in out:
        return [no_label]
    return [no_label]


def _parse_level2_multi_result(
    text: str,
    level2_options: List[Level2Clause],
    ) -> tuple[List[str], bool]:
    """Parse L2 multi-select JSON array. Returns (labels, parsed_ok)."""
    selected = _parse_json_array(text)
    if not selected and "[]" not in (text or ""):
        return ([], False)
    valid_names = {c.name for c in level2_options}
    out: List[str] = []
    seen: set[str] = set()
    for raw_name in selected:
        name = str(raw_name).strip()
        if not name or name in seen or name not in valid_names:
            continue
        seen.add(name)
        out.append(name)
    return (out, True)


class QwenLabelClient:
    """Client for calling Qwen for Level-1 and Level-2 classification."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        l1_sampling: SamplingConfig | None = None,
        l2_sampling: SamplingConfig | None = None,
        l1_max_tokens: int | None = None,
        l2_max_tokens: int | None = None,
    ):
        self._client = OpenAI(
            base_url=base_url or config.QWEN_BASE_URL,
            api_key=api_key or config.QWEN_API_KEY,
        )
        self._model = model or config.QWEN_MODEL
        self._timeout = timeout if timeout is not None else config.QWEN_TIMEOUT
        self._l1_sampling = l1_sampling or sampling_from_mode("nothinking")
        self._l2_sampling = l2_sampling or sampling_from_mode("thinking")
        self._l1_max_tokens = (
            l1_max_tokens if l1_max_tokens is not None else config.ANNOTATE_MAX_TOKENS
        )
        self._l2_max_tokens = (
            l2_max_tokens if l2_max_tokens is not None else config.ANNOTATE_MAX_TOKENS
        )

    def _chat(
        self,
        system: str,
        user: str,
        sampling: SamplingConfig,
        max_tokens: int,
    ) -> str:
        """Send one round of dialogue, return assistant message content."""
        return _chat_once(
            client=self._client,
            model=self._model,
            timeout=self._timeout,
            sampling=sampling,
            max_tokens=max_tokens,
            system=system,
            user=user,
        )

    def classify_level1(
        self,
        sentence: str,
        categories: List[Level1Category],
        heading_context: List[str] | None = None,
    ) -> List[str]:
        """Level-1 multi-select classification. Returns [No_label] if no category matches; raises on unparseable output."""
        system = build_level1_system(categories)
        user = build_level1_user(sentence, heading_context=heading_context)
        def _parser(raw: str) -> List[str] | None:
            parsed, parsed_ok = _parse_level1_result(raw, categories)
            return parsed if parsed_ok else None

        parsed, last_raw = parse_with_retries(
            raw_call=lambda: self._chat(system, user, self._l1_sampling, self._l1_max_tokens),
            parser=_parser,
            attempts=2,
            backoff_base_sec=0.0,
        )
        if parsed is not None:
            return parsed
        raise ValueError(
            "L1 model output parse failed (expected JSON array of labels). "
            f"raw_output={last_raw[:1200]!r}"
        )

    def classify_level2(
        self,
        sentence: str,
        level1_name: str,
        level2_options: List[Level2Clause],
        heading_context: List[str] | None = None,
    ) -> List[str]:
        """Level-2 multi-select within a L1 group: return multiple subclass hits for one sentence; raises on parse failure."""
        if not level2_options:
            return []
        system = build_level2_multi_system()
        user = build_level2_multi_user(
            sentence, level1_name, level2_options, heading_context=heading_context
        )
        def _parser(raw: str) -> List[str] | None:
            result, parsed_ok = _parse_level2_multi_result(raw, level2_options)
            return result if parsed_ok else None

        parsed, last_raw = parse_with_retries(
            raw_call=lambda: self._chat(system, user, self._l2_sampling, self._l2_max_tokens),
            parser=_parser,
            attempts=2,
            backoff_base_sec=0.0,
        )
        if parsed is not None:
            return parsed
        raise ValueError(
            f"L2 model output parse failed for level1={level1_name!r} "
            f"(expected JSON array of level-2 names). raw_output={last_raw[:1200]!r}"
        )


def _chat_once(
    client: OpenAI,
    model: str,
    timeout: float,
    sampling: SamplingConfig,
    max_tokens: int,
    system: str,
    user: str,
) -> str:
    """Shared OpenAI-compatible request function used by annotate/extract/normalize."""
    extra_body = {
        "top_k": sampling.top_k,
        "min_p": sampling.min_p,
        "repetition_penalty": sampling.repetition_penalty,
        "chat_template_kwargs": {"enable_thinking": sampling.enable_thinking},
    }
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            timeout=timeout,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            presence_penalty=sampling.presence_penalty,
            extra_body=extra_body,
        )
    except Exception as err:
        OpenAIChatClient._raise_on_vllm_fatal(err)
    if resp.choices and resp.choices[0].message:
        return (resp.choices[0].message.content or "").strip()
    return ""
