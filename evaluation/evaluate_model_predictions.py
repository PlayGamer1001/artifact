from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


NO_LABEL = "No_label"
EXTRACT_TARGET_LABELS = {
    "Purposes for PI Processing",
    "Scope of PI Collection",
    "Third-Party Recipients of PI",
    "Sources of PI Collected",
    "Scope of PI Sold/Shared/Disclosed",
    "PI Retention Periods/Criteria",
    "Last Updated Date/Effective Date",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare golden labels with model predictions and print three final result groups."
    )
    parser.add_argument(
        "--golden",
        default=str(SCRIPT_DIR / "golden_labels_example.json"),
        help="Path to the golden label JSON file.",
    )
    parser.add_argument(
        "--annotate-dir",
        default=str(PROJECT_ROOT / "output" / "Qwen3.5-27B-FP8" / "annotate" / "answer"),
        help="Directory of annotate answer JSONL files.",
    )
    parser.add_argument(
        "--extract-dir",
        default=str(PROJECT_ROOT / "output" / "Qwen3.5-27B-FP8" / "extract"),
        help="Directory of extract JSONL files.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write metrics as JSON.",
    )
    return parser.parse_args()


def sid_from_source_file(source_file: str, line_index: int) -> str:
    stem = source_file[:-6] if source_file.endswith(".jsonl") else source_file
    return f"{stem.replace('.', '_').lower()}_{line_index + 1:04d}"


def labels_from_annotation(annotation: dict) -> list[str]:
    labels: list[str] = []
    has_no_label = False

    for result in annotation.get("result", []):
        value = result.get("value") or {}

        choices = value.get("choices")
        if isinstance(choices, list) and any(choice == "No Label" for choice in choices):
            has_no_label = True

        taxonomy = value.get("taxonomy")
        if isinstance(taxonomy, list):
            for path in taxonomy:
                if isinstance(path, list) and path and isinstance(path[0], str) and path[0]:
                    labels.append(path[0])

    if has_no_label or not labels:
        return [NO_LABEL]

    deduped: list[str] = []
    seen: set[str] = set()
    for label in labels:
        if label not in seen:
            seen.add(label)
            deduped.append(label)
    return deduped


def final_gold_labels(item: dict) -> list[str]:
    annotations = item.get("annotations", [])
    if not annotations:
        return [NO_LABEL]

    if len(annotations) >= 3:
        return labels_from_annotation(annotations[-1])

    if len(annotations) >= 2:
        first = labels_from_annotation(annotations[0])
        second = labels_from_annotation(annotations[1])
        if first == second:
            return first

    return labels_from_annotation(annotations[0])


def load_golden(path: Path) -> dict[str, set[str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    golden: dict[str, set[str]] = {}
    for item in data:
        sentence_id = (item.get("data") or {}).get("sentence_id")
        if not sentence_id:
            continue
        golden[sentence_id.lower()] = set(final_gold_labels(item))
    return golden


def load_extract_non_empty_labels(extract_dir: Path) -> dict[str, set[str]]:
    non_empty_by_sid: dict[str, set[str]] = defaultdict(set)

    for file_path in sorted(extract_dir.glob("*.jsonl")):
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                sid = sid_from_source_file(record["source_file"], int(record["line_index"]))
                extraction = record.get("extraction") or {}
                for label, payload in extraction.items():
                    if payload:
                        non_empty_by_sid[sid].add(label)

    return non_empty_by_sid


def load_predictions(annotate_dir: Path, extract_dir: Path) -> dict[str, set[str]]:
    extract_non_empty = load_extract_non_empty_labels(extract_dir)
    predictions: dict[str, set[str]] = {}

    for file_path in sorted(annotate_dir.glob("*.jsonl")):
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                sid = sid_from_source_file(record["source_file"], int(record["line_index"]))

                level1 = record.get("level1") or []
                top_is_no_label = ("No_label" in level1) or ("No Label" in level1) or not level1

                level2 = record.get("level2") or {}
                raw_l2: list[str] = []
                if isinstance(level2, dict):
                    for labels in level2.values():
                        if isinstance(labels, list):
                            for label in labels:
                                if isinstance(label, str) and label:
                                    raw_l2.append(label)

                filtered_l2: list[str] = []
                seen: set[str] = set()
                allowed = extract_non_empty.get(sid, set())
                for label in raw_l2:
                    if label in EXTRACT_TARGET_LABELS and label not in allowed:
                        continue
                    if label not in seen:
                        seen.add(label)
                        filtered_l2.append(label)

                if top_is_no_label or not filtered_l2:
                    predictions[sid] = {NO_LABEL}
                else:
                    predictions[sid] = set(filtered_l2)

    return predictions


def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def set_f1(a: set[str], b: set[str]) -> float:
    denom = len(a) + len(b)
    if denom == 0:
        return 1.0
    return 2 * len(a & b) / denom


def macro_micro_f1(samples: Iterable[tuple[set[str], set[str]]]) -> tuple[float, float]:
    pairs = list(samples)
    label_space: set[str] = set()
    for gold, pred in pairs:
        label_space |= gold | pred

    tp_micro = fp_micro = fn_micro = 0
    macro_scores: list[float] = []

    for label in sorted(label_space):
        tp = fp = fn = 0
        for gold, pred in pairs:
            gold_has = label in gold
            pred_has = label in pred
            if gold_has and pred_has:
                tp += 1
            elif (not gold_has) and pred_has:
                fp += 1
            elif gold_has and (not pred_has):
                fn += 1

        tp_micro += tp
        fp_micro += fp
        fn_micro += fn
        denom = 2 * tp + fp + fn
        macro_scores.append((2 * tp / denom) if denom else 0.0)

    micro = 0.0
    if (2 * tp_micro + fp_micro + fn_micro) > 0:
        micro = 2 * tp_micro / (2 * tp_micro + fp_micro + fn_micro)

    macro = sum(macro_scores) / len(macro_scores) if macro_scores else 0.0
    return micro, macro


def sentence_id_to_file_key(sentence_id: str) -> str:
    return sentence_id.rsplit("_", 1)[0] + ".jsonl"


def main() -> None:
    args = parse_args()

    golden = load_golden(Path(args.golden))
    predictions = load_predictions(Path(args.annotate_dir), Path(args.extract_dir))

    common_sids = sorted(set(golden) & set(predictions))
    if not common_sids:
        raise SystemExit("No aligned records found between golden labels and predictions.")

    tp = fp = fn = tn = 0
    step2_pairs: list[tuple[set[str], set[str]]] = []
    overall_pairs: list[tuple[set[str], set[str]]] = []

    for sid in common_sids:
        gold_set = golden[sid]
        pred_set = predictions[sid]

        gold_is_label = any(label != NO_LABEL for label in gold_set)
        pred_is_label = any(label != NO_LABEL for label in pred_set)

        if gold_is_label and pred_is_label:
            tp += 1
        elif (not gold_is_label) and pred_is_label:
            fp += 1
        elif gold_is_label and (not pred_is_label):
            fn += 1
        else:
            tn += 1

        if gold_is_label:
            step2_pairs.append(
                ({label for label in gold_set if label != NO_LABEL}, {label for label in pred_set if label != NO_LABEL})
            )

        overall_pairs.append((gold_set, pred_set))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    step2_n = len(step2_pairs)
    step2_jaccard = sum(jaccard(gold_set, pred_set) for gold_set, pred_set in step2_pairs) / step2_n if step2_n else 0.0
    step2_f1 = sum(set_f1(gold_set, pred_set) for gold_set, pred_set in step2_pairs) / step2_n if step2_n else 0.0

    overall_em = sum(1 for gold_set, pred_set in overall_pairs if gold_set == pred_set) / len(overall_pairs)
    overall_micro_f1, overall_macro_f1 = macro_micro_f1(overall_pairs)

    file_gold: dict[str, set[str]] = defaultdict(set)
    file_pred: dict[str, set[str]] = defaultdict(set)
    for sid in common_sids:
        file_key = sentence_id_to_file_key(sid)
        file_gold[file_key] |= {label for label in golden[sid] if label != NO_LABEL}
        file_pred[file_key] |= {label for label in predictions[sid] if label != NO_LABEL}

    file_keys = sorted(set(file_gold) | set(file_pred))
    file_pairs = [(file_gold[key], file_pred[key]) for key in file_keys]
    file_em = sum(1 for gold_set, pred_set in file_pairs if gold_set == pred_set) / len(file_pairs)
    file_jaccard = sum(jaccard(gold_set, pred_set) for gold_set, pred_set in file_pairs) / len(file_pairs)
    file_micro_f1, _ = macro_micro_f1(file_pairs)

    metrics = {
        "aligned_records": len(common_sids),
        "step1": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": {
                "no_label_to_no_label": tn,
                "no_label_to_label": fp,
                "label_to_no_label": fn,
                "label_to_label": tp,
            },
        },
        "step2": {
            "n": step2_n,
            "jaccard": step2_jaccard,
            "f1": step2_f1,
        },
        "overall": {
            "exact_match": overall_em,
            "micro_f1": overall_micro_f1,
            "macro_f1": overall_macro_f1,
        },
        "file_level": {
            "exact_match": file_em,
            "jaccard": file_jaccard,
            "micro_f1": file_micro_f1,
        },
    }

    print(f"Aligned records: {metrics['aligned_records']}")
    print()
    print("Step1")
    print(f"P/R/F1 = {precision * 100:.2f}% / {recall * 100:.2f}% / {f1 * 100:.2f}%")
    print("Confusion Matrix (Gold x Pred)")
    print(f"No_label -> No_label: {tn}")
    print(f"No_label -> Label: {fp}")
    print(f"Label -> No_label: {fn}")
    print(f"Label -> Label: {tp}")
    print()
    print("Step2")
    print(f"N={step2_n}, Jaccard/F1 = {step2_jaccard * 100:.2f}% / {step2_f1 * 100:.2f}%")
    print()
    print("Overall")
    print(
        f"EM / Micro-F1 / Macro-F1 = {overall_em * 100:.2f}% / "
        f"{overall_micro_f1 * 100:.2f}% / {overall_macro_f1 * 100:.2f}%"
    )
    print()
    print("File-level aggregation")
    print(
        f"EM / Jaccard / Micro-F1 = {file_em * 100:.2f}% / "
        f"{file_jaccard * 100:.2f}% / {file_micro_f1 * 100:.2f}%"
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
