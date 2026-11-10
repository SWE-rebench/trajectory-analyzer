#!/usr/bin/env python3
"""Summarize historical model-consensus agreement and archived parser-4 scores."""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

FIELDS = ("primary_operation", "target_role", "mutation_kind", "read_mode")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def fleiss_kappa(events: list[dict[str, str]], field: str) -> float | None:
    """Unweighted raw-label Fleiss kappa over six-response-complete events."""
    totals: collections.Counter = collections.Counter()
    agreement = []
    for event in events:
        votes = json.loads(event[f"vote_counts_{field}_json"])
        if sum(votes.values()) != 6 or any(not isinstance(v, int) or v < 0 for v in votes.values()):
            raise ValueError("Invalid six-model vote counts")
        totals.update(votes)
        agreement.append((sum(v * v for v in votes.values()) - 6) / 30)
    if not agreement:
        return None
    n = len(agreement)
    pbar = sum(agreement) / n
    pe = sum((v / (6 * n)) ** 2 for v in totals.values())
    return (pbar - pe) / (1 - pe) if pe < 1 else None


def score_events(events: list[dict[str, str]], field: str) -> dict[str, object]:
    """Accuracy and macro-F1 over consensus-resolved supported field scores."""
    selected = [e for e in events if e[f"{field}_supported"].lower() == "true"]
    labels = sorted({e[f"{field}_reference"] for e in selected} | {e[f"{field}_prediction"] for e in selected})

    def calculate(weighted: bool) -> tuple[float | None, float | None]:
        weight = lambda e: float(e["sampling_weight"]) if weighted else 1.0
        denom = sum(weight(e) for e in selected)
        if not denom:
            return None, None
        accuracy = sum(weight(e) for e in selected if e[f"{field}_reference"] == e[f"{field}_prediction"]) / denom
        f1s = []
        for label in labels:
            tp = sum(weight(e) for e in selected if e[f"{field}_reference"] == label and e[f"{field}_prediction"] == label)
            fp = sum(weight(e) for e in selected if e[f"{field}_reference"] != label and e[f"{field}_prediction"] == label)
            fn = sum(weight(e) for e in selected if e[f"{field}_reference"] == label and e[f"{field}_prediction"] != label)
            f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
        return accuracy, sum(f1s) / len(f1s) if f1s else None

    accuracy, macro = calculate(False)
    weighted_accuracy, weighted_macro = calculate(True)
    return {"supported": len(selected), "accuracy": accuracy, "macro_f1": macro,
            "sampling_weighted_accuracy": weighted_accuracy, "sampling_weighted_macro_f1": weighted_macro,
            "macro_f1_labels": labels}


def summarize(consensus: list[dict[str, str]], scores: list[dict[str, str]]) -> dict[str, object]:
    by_id = {e["audit_event_id"]: e for e in consensus}
    if len(by_id) != len(consensus) or len({e['audit_event_id'] for e in scores}) != len(scores):
        raise ValueError("Duplicate audit event ID")
    eligible = [e for e in consensus if e["six_response_eligible"].lower() == "true"]
    for row in scores:
        event = by_id.get(row["audit_event_id"])
        if event is None or event["six_response_eligible"].lower() != "true":
            raise ValueError("Score missing an eligible consensus event")
        if float(row["sampling_weight"]) != float(event["sampling_weight"]):
            raise ValueError("Sampling weight mismatch")
        for field in FIELDS:
            if row[field + "_reference"] != event["consensus_" + field]:
                raise ValueError("Parser score reference differs from consensus")
            correct = row[field + "_reference"] == row[field + "_prediction"] and row[field + "_supported"].lower() == "true"
            if correct != (row[field + "_correct"].lower() == "true"):
                raise ValueError("Parser score correctness mismatch")
    return {"sample_events": len(consensus), "six_response_complete_events": len(eligible),
            "scored_events": len(scores),
            "note": "Agreement is raw-label unweighted Fleiss kappa on six-response-complete events. Parser scores use supplied predictions; model consensus is not human ground truth.",
            "fields": {field: {"fleiss_kappa": fleiss_kappa(eligible, field), **score_events(scores, field)} for field in FIELDS}}


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--consensus", type=Path, required=True, help="consensus and vote-count CSV")
    cli.add_argument("--scores", type=Path, required=True, help="parser predictions and support CSV")
    cli.add_argument("--output", type=Path, help="optional path for the JSON summary")
    args = cli.parse_args()
    summary = summarize(rows(args.consensus), rows(args.scores))
    result = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
