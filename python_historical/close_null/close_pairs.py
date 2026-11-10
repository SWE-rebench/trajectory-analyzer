#!/usr/bin/env python3
"""Recompute fixed Close-1/Close-2 workflow anchors from the parser-4 registry."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

STAGE = Path(__file__).resolve().parent
ROOT = STAGE.parent
LOCK = ROOT / "frozen/PARSER_V4_CLOSE_INPUTS_v1.lock.json"
OUT = STAGE
COHORT = "evaluated_final_v2"
METRICS = [
    "operational_action_count", "exploration_share", "non_test_repository_write_without_observed_failure_count",
    "formal_test_invocation_count", "repeated_unsuccessful_command_count", "unproductive_tail_length",
]
EXPLORE = {"EXPLORE_REPOSITORY", "SEARCH_REPOSITORY", "READ_FILE"}
UNITS = {
    "operational_action_count": "actions/rollout", "exploration_share": "percentage points",
    "non_test_repository_write_without_observed_failure_count": "non-test writes without observed failure/rollout", "formal_test_invocation_count": "contained test invocations/rollout",
    "repeated_unsuccessful_command_count": "repeats/rollout",
    "unproductive_tail_length": "actions/rollout",
}
MULTIPLIER = {m: 100.0 if m == "exploration_share" else 1.0 for m in METRICS}
PRIOR_METRIC_ID = {
    "operational_action_count": "operational_action_count",
    "exploration_share": "exploration_share",
    "non_test_repository_write_without_observed_failure_count": "source_write_count",
    "formal_test_invocation_count": "formal_test_count",
    "repeated_unsuccessful_command_count": "repeated_unsuccessful_command_count",
    "unproductive_tail_length": "unproductive_tail_length",
}
DRAWS = 10_000
# Preserve prior bootstrap and maxT draw keys to isolate parser-metric changes.
BOOT_KEY = "outcome-matched-behavior-bootstrap-v1"
MAXT_KEY = "parser-v2.2.1-close-pair-maxT-v1"  # continuity key preserved intentionally


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(data)


def seed(key: str, *parts: str) -> int:
    return int(hashlib.sha256("|".join((key, *parts)).encode()).hexdigest()[:16], 16)


def describe(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(a.mean()), "standard_deviation": float(a.std(ddof=1)),
        "minimum": float(a.min()), "p10": float(np.quantile(a, .10, method="linear")),
        "q25": float(np.quantile(a, .25, method="linear")),
        "median": float(np.quantile(a, .50, method="linear")),
        "q75": float(np.quantile(a, .75, method="linear")),
        "p90": float(np.quantile(a, .90, method="linear")), "maximum": float(a.max()),
    }


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def main() -> None:
    lock = json.loads(LOCK.read_text())
    if lock["status"] != "frozen_before_downstream_execution" or not lock["execution_authorized_by_user"]:
        raise RuntimeError("Recalculation is not authorized")
    for item in lock["sources"].values():
        path = Path(item["path"])
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"Frozen source hash mismatch: {path}")
    source = {key: Path(item["path"]) for key, item in lock["sources"].items()}

    close1_ids = {r["candidate_pair_id"] for r in rows(source["close1_registry"])}
    close2_ids = {r["candidate_pair_id"] for r in rows(source["close2_registry"])}
    if len(close1_ids) != 65 or len(close2_ids) != 160 or not close1_ids <= close2_ids:
        raise RuntimeError("Close-1/Close-2 registries do not reconcile")
    outcomes = {r["candidate_pair_id"]: r for r in rows(source["pair_outcomes"])}
    pair_tasks: dict[str, list[dict[str, str]]] = defaultdict(list)
    selected_rollouts: set[str] = set()
    for row in rows(source["balanced_pair_task_registry"]):
        if row["candidate_pair_id"] not in close2_ids:
            continue
        pair_tasks[row["candidate_pair_id"]].append(row)
        selected_rollouts.update(json.loads(row["stack_a_rollout_ids_json"]))
        selected_rollouts.update(json.loads(row["stack_b_rollout_ids_json"]))
    if sum(map(len, pair_tasks.values())) != 8_000 or any(len(v) != 50 for v in pair_tasks.values()):
        raise RuntimeError("Expected 160 pairs x 50 tasks")

    # Project the six frozen canonical rollout metrics without coercing nulls.
    metric_records: dict[tuple[str, str], dict[str, Any]] = {}
    columns = [*METRICS, "rollout_id", "semantic_denominator", "parser_id"]
    for batch in pq.ParquetFile(source["canonical_rollout_metrics"]).iter_batches(batch_size=200_000, columns=columns):
        d = batch.to_pydict()
        for i in range(batch.num_rows):
            rid = d["rollout_id"][i]
            if rid not in selected_rollouts:
                continue
            if d["parser_id"][i] != "behavioral-parser-4.0.0+git.4ec73bdbd306":
                raise RuntimeError("Unexpected parser version")
            for metric in METRICS:
                value = d[metric][i]
                metric_records[(rid, metric)] = {
                    "value": None if value is None else float(value),
                    "defined": value is not None,
                    "censored": False,
                    "denominator": d["semantic_denominator"][i] if metric == "exploration_share" else None,
                    "provenance": "canonical_parser_v4_rollout_registry",
                }
    if len(metric_records) != len(selected_rollouts) * len(METRICS):
        raise RuntimeError(f"Canonical metric grid incomplete: {len(metric_records)}/{len(selected_rollouts)*len(METRICS)}")
    anchor_rows = [
        {"rollout_id": rid, "metric_id": metric, **metric_records[(rid, metric)]}
        for rid in sorted(selected_rollouts) for metric in METRICS
    ]

    OUT.mkdir(parents=True, exist_ok=True)
    anchor_path = OUT / "close2_rollout_anchor_metrics_parser_v4_v1.parquet"
    pq.write_table(pa.Table.from_pylist(anchor_rows), anchor_path, compression="zstd")

    estimate_rows: list[dict[str, Any]] = []
    task_diffs: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for pair_id in sorted(close2_ids):
        outcome = outcomes[pair_id]
        for metric in METRICS:
            values_a, values_b, diffs = [], [], []
            defined_a = defined_b = cells_a = cells_b = 0
            for task in pair_tasks[pair_id]:
                ids_a = json.loads(task["stack_a_rollout_ids_json"]); ids_b = json.loads(task["stack_b_rollout_ids_json"])
                va = [metric_records[(rid, metric)]["value"] for rid in ids_a if metric_records[(rid, metric)]["defined"] and metric_records[(rid, metric)]["value"] is not None]
                vb = [metric_records[(rid, metric)]["value"] for rid in ids_b if metric_records[(rid, metric)]["defined"] and metric_records[(rid, metric)]["value"] is not None]
                defined_a += len(va); defined_b += len(vb); cells_a += int(len(va) >= 4); cells_b += int(len(vb) >= 4)
                if len(va) >= 4 and len(vb) >= 4:
                    ma, mb = float(np.mean(va)), float(np.mean(vb))
                    values_a.append(ma); values_b.append(mb); diffs.append(mb - ma)
                    task_diffs[pair_id][metric][task["task_id"]] = mb - ma
            supported = len(diffs) >= 40
            mean_a = mean_b = delta = low = high = None; excludes = False; direction = "unsupported"
            if supported:
                a = np.asarray(diffs, dtype=np.float64)
                mean_a, mean_b, delta = float(np.mean(values_a)), float(np.mean(values_b)), float(a.mean())
                rng = np.random.default_rng(seed(BOOT_KEY, pair_id, metric))
                idx = rng.integers(0, len(a), size=(DRAWS, len(a)), dtype=np.int16)
                boot = a[idx].mean(axis=1)
                low, high = (float(x) for x in np.quantile(boot, [.025, .975], method="linear"))
                excludes = low > 0 or high < 0
                direction = "positive" if delta > 0 else "negative" if delta < 0 else "zero"
            scale = MULTIPLIER[metric]
            estimate_rows.append({
                "candidate_pair_id": pair_id, "in_close_1pp_primary": bool_text(pair_id in close1_ids),
                "in_close_2pp_sensitivity": "true", "stack_a_id": outcome["stack_a_id"], "stack_b_id": outcome["stack_b_id"],
                "model_a": outcome["model_a"], "model_b": outcome["model_b"], "company_a": outcome["company_a"], "company_b": outcome["company_b"],
                "metric_id": metric, "display_unit": UNITS[metric], "task_pool": 50, "paired_defined_tasks": len(diffs),
                "stack_a_task_cells_min4_defined": cells_a, "stack_b_task_cells_min4_defined": cells_b,
                "stack_a_defined_rollouts": defined_a, "stack_b_defined_rollouts": defined_b,
                "support_status": "supported" if supported else "unsupported", "stack_a_mean_native": mean_a, "stack_b_mean_native": mean_b,
                "delta_b_minus_a_native": delta, "ci95_low_native": low, "ci95_high_native": high,
                "delta_b_minus_a_display": None if delta is None else delta * scale,
                "ci95_low_display": None if low is None else low * scale, "ci95_high_display": None if high is None else high * scale,
                "direction": direction, "nominal_ci_excludes_zero": bool_text(excludes),
                "parser_version": "behavioral-parser-4.0.0+git.4ec73bdbd306",
            })

    # Post-hoc within-pair six-anchor familywise sensitivity using shared task sign flips.
    multiplicity_rows = []
    for pair_id in sorted(close2_ids):
        task_order = sorted({task["task_id"] for task in pair_tasks[pair_id]})
        rng = np.random.default_rng(seed(MAXT_KEY, pair_id))
        signs = rng.choice(np.array([-1.0, 1.0]), size=(DRAWS, len(task_order)))
        perm_t_by_metric: dict[str, np.ndarray] = {}; observed_t = {}; raw_p = {}
        for metric in METRICS:
            mapping = task_diffs[pair_id][metric]
            mask = np.array([task in mapping for task in task_order], dtype=bool)
            v = np.array([mapping[task] for task in task_order if task in mapping], dtype=np.float64)
            if len(v) < 40:
                continue
            sd = float(v.std(ddof=1)); obs = float(v.mean())
            observed_t[metric] = math.copysign(math.inf, obs) if sd == 0 and obs != 0 else (0.0 if sd == 0 else obs / (sd / math.sqrt(len(v))))
            means = (signs[:, mask] @ v) / len(v)
            ss = float(np.dot(v, v)); variances = np.maximum((ss - len(v) * means * means) / (len(v) - 1), 0.0)
            ses = np.sqrt(variances / len(v))
            perm_t = np.zeros_like(means)
            np.divide(means, ses, out=perm_t, where=ses > 0)
            degenerate = (ses == 0) & (means != 0)
            perm_t[degenerate] = np.sign(means[degenerate]) * np.inf
            perm_t_by_metric[metric] = np.abs(perm_t)
            raw_p[metric] = (1 + int(np.sum(np.abs(perm_t) >= abs(observed_t[metric])))) / (DRAWS + 1)
        max_t = np.max(np.column_stack([perm_t_by_metric[m] for m in METRICS if m in perm_t_by_metric]), axis=1)
        adjusted = {m: (1 + int(np.sum(max_t >= abs(observed_t[m])))) / (DRAWS + 1) for m in observed_t}
        # Holm adjustment of the same sign-flip p-values as an additional transparent sensitivity.
        ordered = sorted(raw_p, key=raw_p.get); holm = {}; running = 0.0
        for rank, metric in enumerate(ordered):
            candidate = min(1.0, (len(ordered) - rank) * raw_p[metric]); running = max(running, candidate); holm[metric] = running
        for metric in METRICS:
            multiplicity_rows.append({
                "candidate_pair_id": pair_id, "in_close_1pp_primary": bool_text(pair_id in close1_ids), "metric_id": metric,
                "paired_defined_tasks": len(task_diffs[pair_id][metric]), "observed_studentized_statistic": observed_t.get(metric),
                "sign_flip_p_value": raw_p.get(metric), "holm_adjusted_p_value": holm.get(metric), "maxT_adjusted_p_value": adjusted.get(metric),
                "maxT_adjusted_p_lt_0_05": bool_text(adjusted.get(metric, 1.0) < .05), "draws": DRAWS,
                "role": "post_hoc_familywise_sensitivity",
            })

    mult_by_pair = defaultdict(list)
    for row in multiplicity_rows:
        if row["maxT_adjusted_p_value"] is not None:
            mult_by_pair[row["candidate_pair_id"]].append(row)
    estimates_by_pair = defaultdict(list)
    for row in estimate_rows: estimates_by_pair[row["candidate_pair_id"]].append(row)
    profile_rows = []
    for pair_id in sorted(close2_ids):
        erows = estimates_by_pair[pair_id]; supported = [r for r in erows if r["support_status"] == "supported"]
        nominal = sum(r["nominal_ci_excludes_zero"] == "true" for r in supported)
        max_t = sum(r["maxT_adjusted_p_lt_0_05"] == "true" for r in mult_by_pair[pair_id])
        outcome = outcomes[pair_id]; complete = len(supported) == 6
        profile_rows.append({
            "candidate_pair_id": pair_id, "in_close_1pp_primary": bool_text(pair_id in close1_ids), "in_close_2pp_sensitivity": "true",
            "stack_a_id": outcome["stack_a_id"], "stack_b_id": outcome["stack_b_id"], "model_a": outcome["model_a"], "model_b": outcome["model_b"],
            "company_a": outcome["company_a"], "company_b": outcome["company_b"], "solve_delta_pp": float(outcome["paired_delta_b_minus_a"]) * 100,
            "supported_anchor_count": len(supported), "anchors_with_nominal_ci_excluding_zero": nominal,
            "anchors_with_maxT_adjusted_p_lt_0_05": max_t, "all_six_anchors_supported": bool_text(complete),
            "any_nominal_difference": bool_text(complete and nominal >= 1), "at_least_3_nominal_differences": bool_text(complete and nominal >= 3),
            "any_maxT_difference": bool_text(complete and max_t >= 1), "at_least_3_maxT_differences": bool_text(complete and max_t >= 3),
        })

    # Equal-pair native-unit magnitude table over complete Close-1 pairs.
    complete_close1 = [r for r in profile_rows if r["in_close_1pp_primary"] == "true" and r["all_six_anchors_supported"] == "true"]
    complete_ids = {r["candidate_pair_id"] for r in complete_close1}
    magnitude_rows = []
    for metric in METRICS:
        mrows = [r for r in estimate_rows if r["candidate_pair_id"] in complete_ids and r["metric_id"] == metric]
        nominal_rows = [r for r in mrows if r["nominal_ci_excludes_zero"] == "true"]
        max_ids = {r["candidate_pair_id"] for r in multiplicity_rows if r["metric_id"] == metric and r["maxT_adjusted_p_lt_0_05"] == "true"}
        all_stats = describe([abs(float(r["delta_b_minus_a_display"])) for r in mrows])
        sep_stats = describe([abs(float(r["delta_b_minus_a_display"])) for r in nominal_rows]) if nominal_rows else {k: None for k in describe([0.0, 0.0])}
        magnitude_rows.append({
            "metric_id": metric, "display_unit": UNITS[metric], "complete_pairs": len(mrows),
            "nominally_separated_pairs": len(nominal_rows), "nominally_separated_fraction": len(nominal_rows) / len(mrows),
            "maxT_adjusted_pairs": len(max_ids & complete_ids), "maxT_adjusted_fraction": len(max_ids & complete_ids) / len(mrows),
            "positive_nominal_separations": sum(r["direction"] == "positive" for r in nominal_rows),
            "negative_nominal_separations": sum(r["direction"] == "negative" for r in nominal_rows),
            **{f"absolute_delta_all_{k}": v for k, v in all_stats.items()},
            **{f"absolute_delta_nominally_separated_{k}": v for k, v in sep_stats.items()},
        })

    # Bind the revision to the prior parser output and expose every changed pair-anchor row.
    legacy = {(r["candidate_pair_id"], r["metric_id"]): r for r in rows(source["prior_parser_behavior_estimates"])}
    comparison_rows = []
    for row in estimate_rows:
        prior = legacy[(row["candidate_pair_id"], PRIOR_METRIC_ID[row["metric_id"]])]
        old_delta = None if prior["delta_b_minus_a_display"] == "" else float(prior["delta_b_minus_a_display"])
        new_delta = row["delta_b_minus_a_display"]
        comparison_rows.append({
            "candidate_pair_id": row["candidate_pair_id"], "in_close_1pp_primary": row["in_close_1pp_primary"], "metric_id": row["metric_id"],
            "parser_v1_1_delta_display": old_delta, "parser_v4_delta_display": new_delta,
            "delta_of_delta_display": None if old_delta is None or new_delta is None else new_delta - old_delta,
            "parser_v1_1_nominal_separation": prior["nominal_ci_excludes_zero"], "parser_v4_nominal_separation": row["nominal_ci_excludes_zero"],
            "nominal_status_changed": bool_text(prior["nominal_ci_excludes_zero"] != row["nominal_ci_excludes_zero"]),
        })

    estimate_path = OUT / "close_pair_behavior_estimates_parser_v4_v1.csv"
    profile_path = OUT / "close_pair_behavior_profiles_parser_v4_v1.csv"
    mult_path = OUT / "close_pair_multiplicity_sensitivity_parser_v4_v1.csv"
    magnitude_path = OUT / "close1_magnitude_per_anchor_parser_v4_v1.csv"
    compare_path = OUT / "parser_v1_1_to_v4_pair_anchor_changes_v1.csv"
    for path, data in [(estimate_path, estimate_rows), (profile_path, profile_rows), (mult_path, multiplicity_rows), (magnitude_path, magnitude_rows), (compare_path, comparison_rows)]:
        write_csv(path, data)

    close1_nominal = Counter(int(r["anchors_with_nominal_ci_excluding_zero"]) for r in complete_close1)
    close1_maxt = Counter(int(r["anchors_with_maxT_adjusted_p_lt_0_05"]) for r in complete_close1)
    close2_complete = [r for r in profile_rows if r["all_six_anchors_supported"] == "true"]
    change_rows = [r for r in comparison_rows if r["nominal_status_changed"] == "true"]
    summary = {
        "schema_version": "parser-v4-close-pair-recalculation-summary-v1", "status": "complete_pending_validation_gate",
        "population": COHORT, "parser_version": "behavioral-parser-4.0.0+git.4ec73bdbd306", "legacy_exact_zero_used": False,
        "selected_close2_unique_rollouts": len(selected_rollouts),
        "metric_transition": {"all_six_anchors": "direct_from_frozen_canonical_parser_v4_rollout_registry",
            "undefined_values": "preserved_without_zero_coercion",
            "non_test_repository_write_without_observed_failure_count": "non_test_write_without_observed_failure_proxy_not_filesystem_state_change"},
        "close1": {"pairs_total": 65, "complete_pairs": len(complete_close1),
            "nominal_distribution": dict(sorted(close1_nominal.items())), "maxT_distribution": dict(sorted(close1_maxt.items())),
            "any_nominal_pairs": sum(int(r["anchors_with_nominal_ci_excluding_zero"]) >= 1 for r in complete_close1),
            "at_least_3_nominal_pairs": sum(int(r["anchors_with_nominal_ci_excluding_zero"]) >= 3 for r in complete_close1),
            "any_maxT_pairs": sum(int(r["anchors_with_maxT_adjusted_p_lt_0_05"]) >= 1 for r in complete_close1),
            "at_least_3_maxT_pairs": sum(int(r["anchors_with_maxT_adjusted_p_lt_0_05"]) >= 3 for r in complete_close1)},
        "close2": {"pairs_total": 160, "complete_pairs": len(close2_complete),
            "any_nominal_pairs": sum(int(r["anchors_with_nominal_ci_excluding_zero"]) >= 1 for r in close2_complete),
            "at_least_3_nominal_pairs": sum(int(r["anchors_with_nominal_ci_excluding_zero"]) >= 3 for r in close2_complete)},
        "prior_parser_comparison": {"pair_anchor_rows": len(comparison_rows), "nominal_status_changed_rows": len(change_rows),
            "changed_rows_by_metric": dict(Counter(r["metric_id"] for r in change_rows))},
        "multiplicity_role": "post_hoc_familywise_sensitivity_not_primary_preregistered_test",
        "outputs": {}, "inputs_lock": {"path": str(LOCK), "sha256": sha256(LOCK)},
        "software": {"python": "3", "numpy": np.__version__, "pyarrow": pa.__version__},
    }
    for path in [anchor_path, estimate_path, profile_path, mult_path, magnitude_path, compare_path]:
        summary["outputs"][path.name] = {"sha256": sha256(path), "rows": pq.ParquetFile(path).metadata.num_rows if path.suffix == ".parquet" else sum(1 for _ in path.open()) - 1}
    summary_path = OUT / "parser_v4_close_pair_recalculation_summary_v1.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"close1": summary["close1"], "close2": summary["close2"], "prior_parser_comparison": summary["prior_parser_comparison"], "magnitude": magnitude_rows}, indent=2))


if __name__ == "__main__":
    main()
