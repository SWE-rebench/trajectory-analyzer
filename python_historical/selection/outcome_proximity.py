#!/usr/bin/env python3
"""Compute frozen pair-level solve proximity without loading behavior values."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.csv as pacsv

COHORT = "evaluated_final_v2"
BOOTSTRAP_SEED = "outcome-matched-pair-solve-bootstrap-v1"
DRAWS = 10_000
TASKS_PER_PAIR = 50
ROSTER_COLUMNS = ["cohort", "selected", "rollout_id", "outcome"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, data: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(data[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(data)


def percentile(draws: np.ndarray, probability: float) -> float:
    return float(np.quantile(draws, probability, method="linear"))


def main() -> None:
    stage = Path(__file__).resolve().parent
    repo = stage.parents[3]
    pooled = repo / "analysis/evolution_134k/workshop_redesign_v1/corrected_pooled_atlas_v5"
    roster_path = pooled / "cohort_roster_v5.csv"
    frozen = stage / "balanced_registry_v1/frozen"
    pair_path = frozen / "balanced_pair_registry_v1.csv"
    pair_task_path = frozen / "balanced_pair_task_registry_v1.csv"
    registry_lock_path = frozen / "BALANCED_OUTCOME_MATCHED_REGISTRY_v1.lock.json"
    proximity_spec_path = stage / "PAIR_LEVEL_OUTCOME_PROXIMITY_SPEC_v1.yaml"
    proximity_lock_path = frozen / "PAIR_LEVEL_OUTCOME_PROXIMITY_SPEC_v1.lock.json"
    execution_lock_path = frozen / "PAIR_LEVEL_OUTCOME_PROXIMITY_EXECUTION_v1.lock.json"

    execution_lock = json.loads(execution_lock_path.read_text(encoding="utf-8"))
    if execution_lock["status"] != "frozen_before_outcome_access" or execution_lock["outcome_access_authorized"] is not True:
        raise RuntimeError("Outcome proximity execution is not authorized by the frozen lock")
    for relative, expected in execution_lock["sources"].items():
        path = repo / relative
        if sha256(path) != expected:
            raise RuntimeError(f"Execution source hash mismatch: {relative}")

    pairs = read_csv(pair_path)
    pair_tasks = read_csv(pair_task_path)
    if len(pairs) != 1_635 or len(pair_tasks) != 81_750:
        raise RuntimeError("Balanced registry count mismatch")

    outcome_table = pacsv.read_csv(roster_path, convert_options=pacsv.ConvertOptions(include_columns=ROSTER_COLUMNS))
    outcome_by_rollout: dict[str, int] = {}
    for row in outcome_table.to_pylist():
        if row["cohort"] != COHORT or not bool(row["selected"]):
            continue
        value = str(row["outcome"]).strip().lower()
        if value not in {"pass", "fail"}:
            raise RuntimeError(f"Unexpected selected outcome value: {value!r}")
        outcome_by_rollout[row["rollout_id"]] = 1 if value == "pass" else 0
    if len(outcome_by_rollout) != 134_329:
        raise RuntimeError(f"Unexpected outcome mapping size: {len(outcome_by_rollout)}")

    tasks_by_pair: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in pair_tasks:
        tasks_by_pair[row["candidate_pair_id"]].append(row)
    if set(tasks_by_pair) != {row["candidate_pair_id"] for row in pairs} or any(len(rows) != TASKS_PER_PAIR for rows in tasks_by_pair.values()):
        raise RuntimeError("Pair-task grouping mismatch")

    seed_int = int(hashlib.sha256(BOOTSTRAP_SEED.encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed_int)
    bootstrap_indices = rng.integers(0, TASKS_PER_PAIR, size=(DRAWS, TASKS_PER_PAIR), dtype=np.int16)

    output_rows: list[dict[str, Any]] = []
    for pair in sorted(pairs, key=lambda row: row["candidate_pair_id"]):
        pair_id = pair["candidate_pair_id"]
        task_rows = tasks_by_pair[pair_id]
        task_rates_a, task_rates_b = [], []
        for row in task_rows:
            ids_a = json.loads(row["stack_a_rollout_ids_json"])
            ids_b = json.loads(row["stack_b_rollout_ids_json"])
            if len(ids_a) != 5 or len(ids_b) != 5:
                raise RuntimeError(f"Unbalanced rollout IDs: {pair_id} {row['task_id']}")
            try:
                task_rates_a.append(sum(outcome_by_rollout[rid] for rid in ids_a) / 5)
                task_rates_b.append(sum(outcome_by_rollout[rid] for rid in ids_b) / 5)
            except KeyError as exc:
                raise RuntimeError(f"Frozen rollout missing outcome: {exc}") from exc
        a = np.asarray(task_rates_a, dtype=np.float64)
        b = np.asarray(task_rates_b, dtype=np.float64)
        delta_task = b - a
        rate_a, rate_b = float(a.mean()), float(b.mean())
        delta = float(delta_task.mean())
        draws_a = a[bootstrap_indices].mean(axis=1)
        draws_b = b[bootstrap_indices].mean(axis=1)
        draws_delta = draws_b - draws_a
        a_low, a_high = percentile(draws_a, 0.025), percentile(draws_a, 0.975)
        b_low, b_high = percentile(draws_b, 0.025), percentile(draws_b, 0.975)
        d95_low, d95_high = percentile(draws_delta, 0.025), percentile(draws_delta, 0.975)
        d90_low, d90_high = percentile(draws_delta, 0.05), percentile(draws_delta, 0.95)
        paired_nondetection = d95_low <= 0 <= d95_high
        close_2 = abs(delta) <= 0.02 + 1e-12 and paired_nondetection
        close_1 = abs(delta) <= 0.01 + 1e-12 and paired_nondetection
        eq_2 = d90_low >= -0.02 - 1e-12 and d90_high <= 0.02 + 1e-12
        eq_1 = d90_low >= -0.01 - 1e-12 and d90_high <= 0.01 + 1e-12
        overlap_low, overlap_high = max(a_low, b_low), min(a_high, b_high)
        overlap_length = max(0.0, overlap_high - overlap_low)
        narrower_width = min(a_high - a_low, b_high - b_low)
        overlap_fraction = overlap_length / narrower_width if narrower_width > 0 else (1.0 if a_low == b_low and a_high == b_high else 0.0)
        output_rows.append({
            "candidate_pair_id": pair_id,
            "stack_a_id": pair["stack_a_id"],
            "stack_b_id": pair["stack_b_id"],
            "model_a": pair["model_a"],
            "model_b": pair["model_b"],
            "company_a": pair["company_a"],
            "company_b": pair["company_b"],
            "scaffold": pair["scaffold_a"],
            "normalized_reasoning_effort": pair["reasoning_effort_a"],
            "api_mode_proxy": pair["api_mode_proxy_a"],
            "tasks": TASKS_PER_PAIR,
            "trajectories_per_task_per_arm": 5,
            "stack_a_resolved_rate": rate_a,
            "stack_a_ci95_low": a_low,
            "stack_a_ci95_high": a_high,
            "stack_b_resolved_rate": rate_b,
            "stack_b_ci95_low": b_low,
            "stack_b_ci95_high": b_high,
            "paired_delta_b_minus_a": delta,
            "absolute_paired_delta": abs(delta),
            "paired_delta_ci95_low": d95_low,
            "paired_delta_ci95_high": d95_high,
            "paired_delta_ci90_low": d90_low,
            "paired_delta_ci90_high": d90_high,
            "paired_95_interval_includes_zero": str(paired_nondetection).lower(),
            "close_2pp_nondetection": str(close_2).lower(),
            "close_1pp_nondetection": str(close_1).lower(),
            "equivalent_within_2pp": str(eq_2).lower(),
            "equivalent_within_1pp": str(eq_1).lower(),
            "arm_95_intervals_overlap": str(overlap_length > 0 or overlap_low == overlap_high).lower(),
            "arm_95_interval_overlap_length": overlap_length,
            "arm_95_interval_overlap_fraction_of_narrower": overlap_fraction,
        })

    out = stage / "outcome_proximity_v1"
    complete_path = out / "pair_level_solve_proximity_v1.csv"
    close2_path = out / "close_2pp_pair_registry_v1.csv"
    close1_path = out / "close_1pp_pair_registry_v1.csv"
    eq2_path = out / "equivalent_2pp_pair_registry_v1.csv"
    eq1_path = out / "equivalent_1pp_pair_registry_v1.csv"
    summary_path = out / "pair_level_solve_proximity_summary_v1.json"
    write_csv(complete_path, output_rows)
    subsets = {
        close2_path: [row for row in output_rows if row["close_2pp_nondetection"] == "true"],
        close1_path: [row for row in output_rows if row["close_1pp_nondetection"] == "true"],
        eq2_path: [row for row in output_rows if row["equivalent_within_2pp"] == "true"],
        eq1_path: [row for row in output_rows if row["equivalent_within_1pp"] == "true"],
    }
    for path, rows_out in subsets.items():
        write_csv(path, rows_out, list(output_rows[0]))

    def subset_diagnostics(flag: str) -> dict[str, Any]:
        rows_flag = [row for row in output_rows if row[flag] == "true"]
        stacks = {row[field] for row in rows_flag for field in ["stack_a_id", "stack_b_id"]}
        models = {row[field] for row in rows_flag for field in ["model_a", "model_b"]}
        return {
            "pairs": len(rows_flag),
            "unique_stacks": len(stacks),
            "unique_models": len(models),
            "cross_company_pairs": sum(row["company_a"] != row["company_b"] for row in rows_flag),
            "within_company_pairs": sum(row["company_a"] == row["company_b"] for row in rows_flag),
            "explicit_effort_pairs": sum(row["normalized_reasoning_effort"] != "__DEFAULT_OR_UNREPORTED__" for row in rows_flag),
        }

    summary = {
        "schema_version": "pair-level-solve-proximity-summary-v1",
        "status": "outcome_proximity_complete_behavior_not_accessed",
        "population": COHORT,
        "balanced_registry_freeze_id": "balanced_outcome_matched_registry_v1_1635x50x5x2",
        "outcome_proximity_freeze_id": "pair_level_outcome_proximity_v1_close2pp_paired_nondetection",
        "execution_freeze_id": execution_lock["freeze_id"],
        "legacy_exact_0_used": False,
        "behavior_or_event_tables_loaded": False,
        "trajectory_payloads_opened": False,
        "token_or_resource_values_loaded": False,
        "complete_pairs": len(output_rows),
        "paired_95_nondetection_pairs": sum(row["paired_95_interval_includes_zero"] == "true" for row in output_rows),
        "arm_95_interval_overlap_pairs": sum(row["arm_95_intervals_overlap"] == "true" for row in output_rows),
        "subsets": {
            "close_2pp_nondetection": subset_diagnostics("close_2pp_nondetection"),
            "close_1pp_nondetection": subset_diagnostics("close_1pp_nondetection"),
            "equivalent_within_2pp": subset_diagnostics("equivalent_within_2pp"),
            "equivalent_within_1pp": subset_diagnostics("equivalent_within_1pp"),
        },
        "bootstrap": {"draws": DRAWS, "seed": BOOTSTRAP_SEED, "numpy_seed_integer": seed_int, "quantile_method": "linear"},
        "stage_boundary": {
            "behavioral_divergence_computed": False,
            "behavior_values_accessed": False,
            "next_stage_automatically_authorized": False,
        },
        "input_hashes": {
            str(roster_path.relative_to(repo)): sha256(roster_path),
            str(pair_path.relative_to(repo)): sha256(pair_path),
            str(pair_task_path.relative_to(repo)): sha256(pair_task_path),
            str(registry_lock_path.relative_to(repo)): sha256(registry_lock_path),
            str(proximity_spec_path.relative_to(repo)): sha256(proximity_spec_path),
            str(proximity_lock_path.relative_to(repo)): sha256(proximity_lock_path),
            str(execution_lock_path.relative_to(repo)): sha256(execution_lock_path),
            str(Path(__file__).resolve().relative_to(repo)): sha256(Path(__file__).resolve()),
        },
        "software": {"python": "3.13", "numpy": np.__version__, "pyarrow": "25.0.1"},
    }
    output_files = [complete_path, close2_path, close1_path, eq2_path, eq1_path]
    summary["output_hashes"] = {path.name: sha256(path) for path in output_files}
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"complete_pairs": len(output_rows), "paired_95_nondetection_pairs": summary["paired_95_nondetection_pairs"], "arm_95_interval_overlap_pairs": summary["arm_95_interval_overlap_pairs"], "subsets": summary["subsets"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
