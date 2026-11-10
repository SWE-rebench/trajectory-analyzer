#!/usr/bin/env python3
"""Freeze exactly 50 tasks and 5 trajectories per arm for each eligible stack pair."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.csv as pacsv

COHORT = "evaluated_final_v2"
TASK_SEED = "outcome-matched-balanced-tasks-v1"
ROLLOUT_SEED = "outcome-matched-balanced-rollouts-v1"
TASKS_PER_PAIR = 50
ROLLOUTS_PER_ARM = 5
ROSTER_COLUMNS = ["cohort", "selected", "rollout_id", "task_id", "stack_id"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rank(seed: str, *values: str) -> str:
    return hashlib.sha256("|".join([seed, *values]).encode()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, data: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(data[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(data)


def main() -> None:
    stage = Path(__file__).resolve().parent
    repo = stage.parents[3]
    pooled = repo / "analysis/evolution_134k/workshop_redesign_v1/corrected_pooled_atlas_v5"
    roster_path = pooled / "cohort_roster_v5.csv"
    support_path = pooled / "parser_support_v5.csv"
    candidates_path = stage / "candidate_pair_support_v1.csv"
    feasibility_path = stage / "metadata_feasibility_summary_v1.json"
    spec_path = stage / "BALANCED_PAIR_TASK_REGISTRY_SPEC_v1.yaml"

    candidates = read_csv(candidates_path)
    eligible = [
        row for row in candidates
        if row["different_model"] == "true"
        and row["primary_configuration_match"] == "true"
        and row["both_parser_gates_pass"] == "true"
        and int(row["common_tasks_min5_repetitions_each"]) >= TASKS_PER_PAIR
    ]
    if len(eligible) != 1_635:
        raise RuntimeError(f"Expected 1,635 eligible pairs, observed {len(eligible)}")

    table = pacsv.read_csv(roster_path, convert_options=pacsv.ConvertOptions(include_columns=ROSTER_COLUMNS))
    stack_task_rollouts: dict[tuple[str, str], list[str]] = defaultdict(list)
    selected_rows = 0
    for row in table.to_pylist():
        if row["cohort"] == COHORT and bool(row["selected"]):
            selected_rows += 1
            stack_task_rollouts[(row["stack_id"], row["task_id"])].append(row["rollout_id"])
    if selected_rows != 134_329:
        raise RuntimeError(f"Unexpected evaluated-final rows: {selected_rows}")

    selected_rollouts: dict[tuple[str, str], list[str]] = {}
    for key, rollout_ids in stack_task_rollouts.items():
        if len(rollout_ids) >= ROLLOUTS_PER_ARM:
            stack, task = key
            chosen = sorted(rollout_ids, key=lambda rid: (rank(ROLLOUT_SEED, stack, task, rid), rid))[:ROLLOUTS_PER_ARM]
            selected_rollouts[key] = chosen

    pair_rows: list[dict[str, Any]] = []
    pair_task_rows: list[dict[str, Any]] = []
    rollout_view_counts: Counter[str] = Counter()
    selected_unique_tasks = set()
    selected_unique_stacks = set()
    selected_unique_models = set()

    for pair in sorted(eligible, key=lambda row: row["candidate_pair_id"]):
        pair_id = pair["candidate_pair_id"]
        stack_a, stack_b = pair["stack_a_id"], pair["stack_b_id"]
        available_tasks = sorted(
            task for (stack, task) in selected_rollouts
            if stack == stack_a and (stack_b, task) in selected_rollouts
        )
        if len(available_tasks) != int(pair["common_tasks_min5_repetitions_each"]):
            raise RuntimeError(f"Candidate/roster task reconciliation failed: {pair_id}")
        chosen_tasks = sorted(
            available_tasks,
            key=lambda task: (rank(TASK_SEED, pair_id, task), task),
        )[:TASKS_PER_PAIR]
        if len(chosen_tasks) != TASKS_PER_PAIR:
            raise RuntimeError(f"Incomplete task selection: {pair_id}")
        task_set_hash = hashlib.sha256("\n".join(sorted(chosen_tasks)).encode()).hexdigest()
        pair_rows.append({
            **pair,
            "available_common_tasks_min5": len(available_tasks),
            "selected_balanced_tasks": TASKS_PER_PAIR,
            "selected_trajectories_per_task_per_arm": ROLLOUTS_PER_ARM,
            "selected_rollout_views": TASKS_PER_PAIR * ROLLOUTS_PER_ARM * 2,
            "selected_task_set_sha256": task_set_hash,
            "task_selection_seed": TASK_SEED,
            "rollout_selection_seed": ROLLOUT_SEED,
        })
        selected_unique_stacks.update([stack_a, stack_b])
        selected_unique_models.update([pair["model_a"], pair["model_b"]])
        for task in chosen_tasks:
            rollouts_a = selected_rollouts[(stack_a, task)]
            rollouts_b = selected_rollouts[(stack_b, task)]
            for rollout_id in [*rollouts_a, *rollouts_b]:
                rollout_view_counts[rollout_id] += 1
            selected_unique_tasks.add(task)
            pair_task_rows.append({
                "candidate_pair_id": pair_id,
                "task_id": task,
                "stack_a_id": stack_a,
                "stack_b_id": stack_b,
                "model_a": pair["model_a"],
                "model_b": pair["model_b"],
                "company_a": pair["company_a"],
                "company_b": pair["company_b"],
                "scaffold": pair["scaffold_a"],
                "normalized_reasoning_effort": pair["reasoning_effort_a"],
                "api_mode_proxy": pair["api_mode_proxy_a"],
                "task_selection_hash_rank": rank(TASK_SEED, pair_id, task),
                "stack_a_rollout_ids_json": json.dumps(rollouts_a, separators=(",", ":")),
                "stack_b_rollout_ids_json": json.dumps(rollouts_b, separators=(",", ":")),
                "stack_a_rollout_set_sha256": hashlib.sha256("\n".join(sorted(rollouts_a)).encode()).hexdigest(),
                "stack_b_rollout_set_sha256": hashlib.sha256("\n".join(sorted(rollouts_b)).encode()).hexdigest(),
                "rollouts_per_arm": ROLLOUTS_PER_ARM,
            })

    expected_pair_tasks = len(eligible) * TASKS_PER_PAIR
    expected_views = expected_pair_tasks * ROLLOUTS_PER_ARM * 2
    if len(pair_task_rows) != expected_pair_tasks or sum(rollout_view_counts.values()) != expected_views:
        raise RuntimeError("Balanced registry count closure failed")

    frozen = stage / "balanced_registry_v1/frozen"
    pair_path = frozen / "balanced_pair_registry_v1.csv"
    pair_task_path = frozen / "balanced_pair_task_registry_v1.csv"
    summary_path = frozen / "balanced_registry_summary_v1.json"
    lock_path = frozen / "BALANCED_OUTCOME_MATCHED_REGISTRY_v1.lock.json"
    write_csv(pair_path, pair_rows)
    write_csv(pair_task_path, pair_task_rows)

    explicit_effort_pairs = sum(row["reasoning_effort_a"] != "__DEFAULT_OR_UNREPORTED__" for row in eligible)
    summary = {
        "schema_version": "balanced-outcome-matched-registry-summary-v1",
        "status": "balanced_metadata_registry_frozen_no_outcome_or_behavior_access",
        "population": COHORT,
        "legacy_exact_0_used": False,
        "outcome_columns_loaded": False,
        "behavior_or_event_tables_loaded": False,
        "trajectory_payloads_opened": False,
        "token_or_resource_values_loaded": False,
        "counts": {
            "eligible_pairs": len(pair_rows),
            "pair_task_rows": len(pair_task_rows),
            "rollout_views": sum(rollout_view_counts.values()),
            "unique_selected_rollouts": len(rollout_view_counts),
            "unique_selected_tasks": len(selected_unique_tasks),
            "unique_selected_stacks": len(selected_unique_stacks),
            "unique_selected_models": len(selected_unique_models),
            "cross_company_pairs": sum(row["company_a"] != row["company_b"] for row in eligible),
            "within_company_pairs": sum(row["company_a"] == row["company_b"] for row in eligible),
            "default_or_unreported_effort_pairs": len(eligible) - explicit_effort_pairs,
            "explicit_effort_pairs": explicit_effort_pairs,
            "maximum_rollout_view_reuse": max(rollout_view_counts.values()),
        },
        "balanced_design": {
            "tasks_per_pair": TASKS_PER_PAIR,
            "trajectories_per_task_per_arm": ROLLOUTS_PER_ARM,
            "rollout_views_per_pair": TASKS_PER_PAIR * ROLLOUTS_PER_ARM * 2,
            "same_tasks_within_pair": True,
            "same_tasks_across_pairs_required": False,
            "rollout_reuse_across_pair_views": True,
        },
        "stage_boundary": {
            "outcome_proximity_definition_frozen": False,
            "outcome_values_accessed": False,
            "behavior_thresholds_frozen": False,
            "behavior_values_accessed": False,
            "weighting_estimand_frozen": False,
            "analysis_execution_authorized": False,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    source_paths = [
        roster_path, support_path, candidates_path, feasibility_path, spec_path,
        Path(__file__).resolve(),
    ]
    outputs = [pair_path, pair_task_path, summary_path]
    lock = {
        "schema_version": "balanced-outcome-matched-registry-lock-v1",
        "freeze_id": "balanced_outcome_matched_registry_v1_1635x50x5x2",
        "status": "frozen_before_outcome_or_behavior_access",
        "population": COHORT,
        "legacy_exact_0_allowed": False,
        "selection_used_outcome": False,
        "selection_used_behavior": False,
        "selection_used_tokens_or_resources": False,
        "pair_count": len(pair_rows),
        "tasks_per_pair": TASKS_PER_PAIR,
        "trajectories_per_task_per_arm": ROLLOUTS_PER_ARM,
        "pair_task_rows": len(pair_task_rows),
        "rollout_views": sum(rollout_view_counts.values()),
        "sources": {str(path.relative_to(repo)): sha256(path) for path in source_paths},
        "outputs": {str(path.relative_to(repo)): sha256(path) for path in outputs},
        "software": {"python": "3.13", "pyarrow": "25.0.1"},
    }
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"freeze_id": lock["freeze_id"], **summary["counts"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
