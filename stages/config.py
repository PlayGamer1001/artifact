# -*- coding: utf-8 -*-
"""Configuration: paths, Qwen API parameters, output directories, and per-stage settings."""

from pathlib import Path

# Project root (directory containing this config file)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Clause taxonomy JSON (Level-1 categories + Level-2 clauses)
CLAUSE_JSON_PATH: Path = _PROJECT_ROOT / "stages" / "annotate" / "clause_labels.json"

# PP_jsonl root (traversed recursively for .jsonl input files)
PP_JSONL_ROOT: Path = _PROJECT_ROOT / "data" / "jsonl"

# Output base directory (pipeline stages write under this)
_OUTPUT_BASE = _PROJECT_ROOT
OUTPUT_DIR: Path = _OUTPUT_BASE / "output"
L1_STAGE_DIRNAME: str = "_stage1"

# Whether to skip empty input lines
SKIP_EMPTY_LINES: bool = True

# Minimum sentence length in characters (0 = no limit)
MIN_SENTENCE_LENGTH: int = 0

# Qwen API (OpenAI-compatible)
QWEN_BASE_URL: str = "http://127.0.0.1:8000/v1"
QWEN_API_KEY: str = "none"
QWEN_MODEL: str = "Qwen3.5-27B-FP8"

# Request timeout in seconds
QWEN_TIMEOUT: float = 120.0

# Per-stage max_tokens for chat.completions
ANNOTATE_MAX_TOKENS: int = 512
EXTRACT_MAX_TOKENS: int = 1024
NORMALIZE_MAX_TOKENS: int = 512

# Sampling parameters by mode
NOTHINKING_ENABLE_THINKING: bool = False
NOTHINKING_TEMPERATURE: float = 0.7
NOTHINKING_TOP_P: float = 0.8
NOTHINKING_TOP_K: int = 20
NOTHINKING_MIN_P: float = 0.0
NOTHINKING_PRESENCE_PENALTY: float = 1.5
NOTHINKING_REPETITION_PENALTY: float = 1.0

THINKING_ENABLE_THINKING: bool = True
THINKING_TEMPERATURE: float = 1.0
THINKING_TOP_P: float = 0.95
THINKING_TOP_K: int = 20
THINKING_MIN_P: float = 0.0
THINKING_PRESENCE_PENALTY: float = 1.5
THINKING_REPETITION_PENALTY: float = 1.0

# Label used when a sentence does not match any Level-1 category
NO_LABEL = "No_label"

# Annotate stage concurrency
L1_CONCURRENCY = 32
L2_CONCURRENCY = 32

# Annotate stage: files per batch (file-level scheduling, not request-level)
FILES_PER_BATCH: int = 1276

# Extract stage
EXTRACT_MODE: str = "nothinking"
EXTRACT_CONCURRENCY: int = 32
EXTRACT_FILES_PER_BATCH: int = 1276

# Normalize stage
NORMALIZE_MODE: str = "nothinking"
NORMALIZE_CONCURRENCY: int = 32
NORMALIZE_FILES_PER_BATCH: int = 1276

# Process stage
PROCESS_INPUT_DIR: Path = _PROJECT_ROOT / "data" / "input"
PROCESS_CONCURRENCY: int = 6
PROCESS_FILES_PER_BATCH: int = 10
