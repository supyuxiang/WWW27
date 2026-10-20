"""Accuracy, macro-F1, calibration, and loop-cost summaries."""

from __future__ import annotations


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def binary_counts(gold: list[str], pred: list[str], label: str) -> tuple[int, int, int]:
    if len(gold) != len(pred):
        raise ValueError("gold and pred must have the same length")
    tp = fp = fn = 0
    for truth, guess in zip(gold, pred):
        if guess == label and truth == label:
            tp += 1
        elif guess == label and truth != label:
            fp += 1
        elif truth == label and guess != label:
            fn += 1
    return tp, fp, fn


def f1_for_label(gold: list[str], pred: list[str], label: str) -> float:
    tp, fp, fn = binary_counts(gold, pred, label)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return _safe_div(2 * precision * recall, precision + recall)


def macro_f1(gold: list[str], pred: list[str], labels: tuple[str, ...] = ("real", "fake")) -> float:
    if not gold:
        return 0.0
    return sum(f1_for_label(gold, pred, label) for label in labels) / len(labels)


def accuracy(gold: list[str], pred: list[str]) -> float:
    if len(gold) != len(pred):
        raise ValueError("gold and pred must have the same length")
    if not gold:
        return 0.0
    correct = sum(truth == guess for truth, guess in zip(gold, pred))
    return correct / len(gold)


def expected_calibration_error(
    confidences: list[float],
    correct: list[bool],
    n_bins: int = 10,
) -> float | None:
    """Confidence ECE over rows that expose a judge distribution.

    Rows without `judge_probs` are excluded, so `n_calibration` in the
    summary can be smaller than `n`. This reports calibration; it does not
    fit a temperature.
    """
    if not confidences or len(confidences) != len(correct):
        return None
    bins = [[] for _ in range(n_bins)]
    for confidence, flag in zip(confidences, correct):
        index = min(n_bins - 1, max(0, int(confidence * n_bins)))
        bins[index].append((confidence, flag))
    total = len(confidences)
    error = 0.0
    for bucket in bins:
        if not bucket:
            continue
        mean_conf = sum(item[0] for item in bucket) / len(bucket)
        mean_acc = sum(1.0 if item[1] else 0.0 for item in bucket) / len(bucket)
        error += (len(bucket) / total) * abs(mean_acc - mean_conf)
    return error


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def usable_rows(rows: list[dict]) -> list[dict]:
    kept = []
    for row in rows:
        if row.get("error"):
            continue
        if row.get("prediction") not in {"real", "fake"}:
            continue
        if row.get("gold") not in {"real", "fake"}:
            continue
        kept.append(row)
    return kept


def summarize(rows: list[dict]) -> dict:
    valid = usable_rows(rows)
    domains = sorted({row.get("domain") or "unknown" for row in rows})
    summary = _flat_summary(valid, rows)
    summary["by_domain"] = {}
    for domain in domains:
        domain_rows = [row for row in rows if (row.get("domain") or "unknown") == domain]
        summary["by_domain"][domain] = _flat_summary(usable_rows(domain_rows), domain_rows)
    return summary


def _flat_summary(valid: list[dict], all_rows: list[dict]) -> dict:
    gold = [row["gold"] for row in valid]
    pred = [row["prediction"] for row in valid]
    confidences = []
    correct_flags = []
    for row in valid:
        probs = row.get("judge_probs") or {}
        if probs:
            confidences.append(float(max(probs.values())))
            correct_flags.append(row["prediction"] == row["gold"])
    n_errors = 0
    if all_rows:
        n_errors = sum(
            1
            for row in all_rows
            if row.get("error") or row.get("prediction") not in {"real", "fake"}
        )
    return {
        "n": len(valid),
        "n_errors": n_errors,
        "accuracy": accuracy(gold, pred),
        "macro_f1": macro_f1(gold, pred),
        "calls_per_item": _mean([float(row.get("search_calls") or 0) for row in valid]),
        "rounds_per_item": _mean([float(row.get("retrieval_rounds") or 0) for row in valid]),
        "stop_gate_rate": _mean([1.0 if row.get("stop_reason") == "gate" else 0.0 for row in valid]),
        "stop_budget_rate": _mean([1.0 if row.get("stop_reason") == "budget" else 0.0 for row in valid]),
        "forward_passes": _mean([float(row.get("forward_passes") or 0) for row in valid]),
        "input_tokens": _mean([float(row.get("input_tokens") or 0) for row in valid]),
        "output_tokens": _mean([float(row.get("output_tokens") or 0) for row in valid]),
        "latency_s": _mean([float(row.get("latency_s") or 0) for row in valid]),
        "parse_failures": _mean([float(row.get("parse_failures") or 0) for row in valid]),
        "fallbacks": _mean([float(row.get("fallbacks") or 0) for row in valid]),
        "retrieval_rate": _mean([1.0 if row.get("retrieved") else 0.0 for row in valid]),
        "n_calibration": len(confidences),
        "ece": expected_calibration_error(confidences, correct_flags),
    }
