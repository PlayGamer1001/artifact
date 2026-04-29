# Artifact

A four-stage privacy policy clause annotation pipeline using LLM classification (default Qwen).

**Python**: 3.12+

The code was verified and run on a single RTX PRO 6000 GPU.

## Included Data

This anonymized artifact includes a 100-policy processed sample:

- `data/jsonl/`: 100 JSONL privacy-policy sentence files, 5749 sentence records in total.
- `evaluation/golden_labels_example.json`: 5749 sanitized golden-label records aligned with the included JSONL sentence IDs.
- No model outputs are included. Generated outputs are written under `output/` and are ignored by git.
- Raw privacy policy files are not included. `data/input/` is only a placeholder for users who want to run the optional processing stage on their own raw files.

## Stage Overview

| Stage | Description |
|-------|-------------|
| `process` | Optional: convert raw HTML/DOCX/PDF/TXT privacy policies to JSONL sentence records |
| `annotate` | Two-phase L1/L2 classification via LLM |
| `extract` | Structured field extraction for target privacy labels |
| `normalize` | Map extracted values to a unified taxonomy |
| `evaluate` | Compare LLM predictions against golden labels |

## Deployment

### 1. Environment setup

**Model setup** — follow [Qwen3.5-27B-FP8 on HuggingFace](https://huggingface.co/Qwen/Qwen3.5-27B-FP8) to download the model and configure vLLM.

**Install dependencies:**

```bash
pip install -r requirements.txt
```

### 2. Model serving

**Start server:**

```bash
vllm serve /path/to/Qwen3.5-27B-FP8 \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name Qwen3.5-27B-FP8 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 32768 \
  --enable-prefix-caching \
  --reasoning-parser qwen3
```

### 3. Configuration

Update `stages/config.py` or pass CLI arguments to override defaults:

```python
QWEN_BASE_URL = "http://127.0.0.1:8000/v1"
QWEN_MODEL = "Qwen3.5-27B-FP8"
```

### 4. Run the pipeline on the included sample

The included artifact already contains preprocessed JSONL files, so the normal reviewer workflow starts from `annotate`:

```bash
python main.py --target-dir sample_run --annotate --extract --normalize
```

Run stages individually if needed:

```bash
python main.py --target-dir sample_run --annotate
python main.py --target-dir sample_run --extract
python main.py --target-dir sample_run --normalize
```

### Optional: process raw files

The `process` stage is only needed if you add your own raw privacy policy files to `data/input/`:

```bash
python main.py --target-dir sample_run --process
```

## Key Arguments

| Argument | Description |
|---|---|
| `--target-dir` | **(required)** Run output root directory name |
| `--process` | Run optional process stage on raw files in `data/input/` |
| `--annotate` | Run annotate stage (L1/L2 classification) |
| `--extract` | Run extract stage (structured extraction) |
| `--normalize` | Run normalize stage (taxonomy normalization) |
| `--pp-jsonl` | Override input JSONL root (default: `data/jsonl`) |
| `--clause-json` | Override clause taxonomy JSON path |
| `--output-dir` | Override output base directory |
| `--process-input-dir` | Override raw input files directory |
| `--base-url` | API base URL (default: `http://127.0.0.1:8000/v1`) |
| `--model` | Model name (default: `Qwen3.5-27B-FP8`) |

## Output Layout

```
data/
  input/                     # Optional raw privacy policy files; not included in this artifact
  jsonl/                     # Included JSONL sentence records for the 100-policy sample

<output-dir>/
  <target-dir>/
    run_errors.jsonl         # Fatal errors JSONL
    process/
      evaluation/            # Language detection + token distribution, if --process is run
    annotate/
      answer/                # Final L1/L2 annotations
      _stage1/               # L1 intermediate results
    extract/                 # Structured extraction results
    normalize/               # Taxonomy-normalized results
```

## Evaluation Module

After generating `annotate/answer` outputs, compare LLM predictions against the sanitized golden labels:

```bash
python evaluation/evaluate_model_predictions.py \
  --golden evaluation/golden_labels_example.json \
  --annotate-dir <output-dir>/<target-dir>/annotate/answer \
  --output evaluation/metrics.json
```

**Metrics computed:**
- **binary_passage**: Binary P/R/F1 (label vs No_label)
- **multi_label_passage**: Jaccard/Micro-F1/Macro-F1 over aligned sentence records
- **multi_label_document**: Jaccard/Micro-F1/Macro-F1 aggregated by source file

**Note**: `evaluation/golden_labels_example.json` contains the sanitized golden labels for the included 100-policy sample.

## Quick Check

```bash
python main.py --help
python evaluation/evaluate_model_predictions.py --help
```
