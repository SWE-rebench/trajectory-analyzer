#!/usr/bin/env python3
"""Metadata-only feasibility audit for outcome-matched behavioral divergence."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.csv as pacsv

COHORT = "evaluated_final_v2"
REPETITION_VIEWS = [1, 2, 3, 5]
ROSTER_COLUMNS = [
    "cohort", "selected", "rollout_id", "task_id", "stack_id", "canonical_model",
    "company", "scaffold", "normalized_reasoning_effort", "api_mode_proxy",
    "parser_format_id", "action_format", "config_hash", "non_effort_config_signature",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, data: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    fields = fields or list(data[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(data)


def describe(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(values), "min": min(values), "median": statistics.median(values),
        "mean": sum(values) / len(values), "max": max(values),
    }


def main() -> None:
    stage = Path(__file__).resolve().parent
    repo = stage.parents[3]
    pooled = repo / "analysis/evolution_134k/workshop_redesign_v1/corrected_pooled_atlas_v5"
    roster_path = pooled / "cohort_roster_v5.csv"
    support_path = pooled / "parser_support_v5.csv"
    spec_path = stage / "METADATA_FEASIBILITY_SPEC_v1.yaml"

    table = pacsv.read_csv(roster_path, convert_options=pacsv.ConvertOptions(include_columns=ROSTER_COLUMNS))
    selected = [
        row for row in table.to_pylist()
        if row["cohort"] == COHORT and bool(row["selected"])
    ]
    if len(selected) != 134_329:
        raise RuntimeError(f"Unexpected evaluated-final selected rows: {len(selected)}")
    if len({row["rollout_id"] for row in selected}) != len(selected):
        raise RuntimeError("Selected rollout IDs are not unique")

    support_rows = list(csv.DictReader(support_path.open(newline="", encoding="utf-8")))
    support_by_stack: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in support_rows:
        if row["cohort"] == COHORT:
            support_by_stack[row["stack_id"]].append(row)

    task_counts: dict[str, Counter[str]] = defaultdict(Counter)
    metadata_values: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    stack_rollouts: Counter[str] = Counter()
    for row in selected:
        stack = row["stack_id"]
        task_counts[stack][row["task_id"]] += 1
        stack_rollouts[stack] += 1
        for field in [
            "canonical_model", "company", "scaffold", "normalized_reasoning_effort",
            "api_mode_proxy", "parser_format_id", "action_format", "config_hash",
            "non_effort_config_signature",
        ]:
            metadata_values[stack][field].add(str(row[field] or ""))

    stacks = sorted(task_counts)
    if len(stacks) != 125:
        raise RuntimeError(f"Unexpected evaluated-final stack count: {len(stacks)}")
    for stack in stacks:
        for field in ["canonical_model", "company", "scaffold", "normalized_reasoning_effort", "api_mode_proxy"]:
            if len(metadata_values[stack][field]) != 1:
                raise RuntimeError(f"Stack metadata not unique: {stack} {field} {metadata_values[stack][field]}")
        if stack not in support_by_stack:
            raise RuntimeError(f"Missing parser support for stack: {stack}")

    def one(stack: str, field: str) -> str:
        return next(iter(metadata_values[stack][field]))

    stack_gate = {
        stack: all(row["support_gate_pass"] == "True" for row in support_by_stack[stack])
        for stack in stacks
    }
    stack_rows: list[dict[str, Any]] = []
    for stack in stacks:
        counts = list(task_counts[stack].values())
        stack_rows.append({
            "stack_id": stack,
            "canonical_model": one(stack, "canonical_model"),
            "company": one(stack, "company"),
            "scaffold": one(stack, "scaffold"),
            "normalized_reasoning_effort": one(stack, "normalized_reasoning_effort"),
            "api_mode_proxy": one(stack, "api_mode_proxy"),
            "parser_format_ids": ";".join(sorted(metadata_values[stack]["parser_format_id"])),
            "action_formats": ";".join(sorted(metadata_values[stack]["action_format"])),
            "config_hash_count": len(metadata_values[stack]["config_hash"]),
            "non_effort_config_signature_count": len(metadata_values[stack]["non_effort_config_signature"]),
            "selected_rollouts": stack_rollouts[stack],
            "tasks_any_repetition": len(task_counts[stack]),
            "tasks_min2_repetitions": sum(n >= 2 for n in counts),
            "tasks_min3_repetitions": sum(n >= 3 for n in counts),
            "tasks_min5_repetitions": sum(n >= 5 for n in counts),
            "min_repetitions_per_task": min(counts),
            "median_repetitions_per_task": statistics.median(counts),
            "max_repetitions_per_task": max(counts),
            "evaluated_final_parser_gate_pass": str(stack_gate[stack]).lower(),
        })

    pair_rows: list[dict[str, Any]] = []
    primary_pair_tasks: dict[str, list[str]] = {}
    for stack_a, stack_b in itertools.combinations(stacks, 2):
        model_a, model_b = one(stack_a, "canonical_model"), one(stack_b, "canonical_model")
        common = set(task_counts[stack_a]) & set(task_counts[stack_b])
        common_by_rep = {
            threshold: [
                task for task in common
                if task_counts[stack_a][task] >= threshold and task_counts[stack_b][task] >= threshold
            ]
            for threshold in REPETITION_VIEWS
        }
        same_scaffold = one(stack_a, "scaffold") == one(stack_b, "scaffold")
        same_effort = one(stack_a, "normalized_reasoning_effort") == one(stack_b, "normalized_reasoning_effort")
        same_api = one(stack_a, "api_mode_proxy") == one(stack_b, "api_mode_proxy")
        different_model = model_a != model_b
        config_match = same_scaffold and same_effort and same_api
        parser_pass = stack_gate[stack_a] and stack_gate[stack_b]
        eligible = different_model and config_match and parser_pass and len(common_by_rep[3]) >= 20
        pair_id = "pair_" + hashlib.sha256(f"{stack_a}|{stack_b}".encode()).hexdigest()[:16]
        row = {
            "candidate_pair_id": pair_id,
            "stack_a_id": stack_a,
            "stack_b_id": stack_b,
            "model_a": model_a,
            "model_b": model_b,
            "company_a": one(stack_a, "company"),
            "company_b": one(stack_b, "company"),
            "scaffold_a": one(stack_a, "scaffold"),
            "scaffold_b": one(stack_b, "scaffold"),
            "reasoning_effort_a": one(stack_a, "normalized_reasoning_effort"),
            "reasoning_effort_b": one(stack_b, "normalized_reasoning_effort"),
            "api_mode_proxy_a": one(stack_a, "api_mode_proxy"),
            "api_mode_proxy_b": one(stack_b, "api_mode_proxy"),
            "different_model": str(different_model).lower(),
            "same_scaffold": str(same_scaffold).lower(),
            "same_reasoning_effort": str(same_effort).lower(),
            "same_api_mode_proxy": str(same_api).lower(),
            "primary_configuration_match": str(config_match).lower(),
            "both_parser_gates_pass": str(parser_pass).lower(),
            "common_tasks_any_repetition": len(common_by_rep[1]),
            "common_tasks_min2_repetitions_each": len(common_by_rep[2]),
            "common_tasks_min3_repetitions_each": len(common_by_rep[3]),
            "common_tasks_min5_repetitions_each": len(common_by_rep[5]),
            "primary_metadata_eligible": str(eligible).lower(),
        }
        pair_rows.append(row)
        if eligible:
            primary_pair_tasks[pair_id] = sorted(common_by_rep[3])

    primary_rows = [row for row in pair_rows if row["primary_metadata_eligible"] == "true"]
    primary_ids = {row["candidate_pair_id"] for row in primary_rows}
    primary_unique_tasks = {task for pair_id in primary_ids for task in primary_pair_tasks[pair_id]}
    primary_pair_task_cells = sum(len(primary_pair_tasks[pair_id]) for pair_id in primary_ids)
    primary_stacks = {row[field] for row in primary_rows for field in ["stack_a_id", "stack_b_id"]}
    primary_models = {row[field] for row in primary_rows for field in ["model_a", "model_b"]}

    def eligible_count(filter_kind: str, threshold: int) -> int:
        def base(row: dict[str, Any]) -> bool:
            if filter_kind == "primary_config":
                return row["different_model"] == "true" and row["primary_configuration_match"] == "true" and row["both_parser_gates_pass"] == "true"
            if filter_kind == "different_model":
                return row["different_model"] == "true" and row["both_parser_gates_pass"] == "true"
            if filter_kind == "all_stacks":
                return row["both_parser_gates_pass"] == "true"
            raise ValueError(filter_kind)
        return sum(base(row) and int(row[f"common_tasks_min{threshold}_repetitions_each"] if threshold > 1 else row["common_tasks_any_repetition"]) >= 20 for row in pair_rows)

    support_grid = {
        kind: {f"min{threshold}_repetitions_each": eligible_count(kind, threshold) for threshold in REPETITION_VIEWS}
        for kind in ["primary_config", "different_model", "all_stacks"]
    }
    candidate_config_rows = [
        row for row in pair_rows
        if row["different_model"] == "true" and row["primary_configuration_match"] == "true" and row["both_parser_gates_pass"] == "true"
    ]
    pair_common_n3 = [int(row["common_tasks_min3_repetitions_each"]) for row in candidate_config_rows]

    summary = {
        "schema_version": "outcome-matched-behavior-metadata-feasibility-v1",
        "status": "metadata_feasibility_complete_pair_registry_not_frozen",
        "population": COHORT,
        "legacy_exact_0_used": False,
        "outcome_columns_loaded": False,
        "behavior_or_event_tables_loaded": False,
        "trajectory_payloads_opened": False,
        "counts": {
            "selected_rollouts": len(selected),
            "unique_tasks": len({row["task_id"] for row in selected}),
            "exact_stacks": len(stacks),
            "canonical_models": len({one(stack, "canonical_model") for stack in stacks}),
            "parser_gate_passing_stacks": sum(stack_gate.values()),
            "all_unordered_stack_pairs": len(pair_rows),
            "different_model_pairs": sum(row["different_model"] == "true" for row in pair_rows),
            "primary_configuration_matched_gate_passing_pairs_before_task_gate": len(candidate_config_rows),
            "primary_metadata_eligible_pairs": len(primary_rows),
            "primary_eligible_unique_stacks": len(primary_stacks),
            "primary_eligible_unique_models": len(primary_models),
            "primary_eligible_unique_tasks": len(primary_unique_tasks),
            "primary_eligible_pair_task_cells_min3": primary_pair_task_cells,
        },
        "eligible_pair_counts_at_20_common_tasks": support_grid,
        "primary_configuration_pair_common_task_distribution_min3": describe(pair_common_n3),
        "primary_eligible_pair_common_task_distribution_min3": describe([
            int(row["common_tasks_min3_repetitions_each"]) for row in primary_rows
        ]),
        "stage_boundary": {
            "pair_registry_frozen": False,
            "outcome_proximity_computed": False,
            "behavioral_difference_computed": False,
            "next_stage_automatically_authorized": False,
        },
        "input_hashes": {
            str(roster_path.relative_to(repo)): sha256(roster_path),
            str(support_path.relative_to(repo)): sha256(support_path),
            str(spec_path.relative_to(repo)): sha256(spec_path),
            str(Path(__file__).resolve().relative_to(repo)): sha256(Path(__file__).resolve()),
        },
        "software": {"python": "3.13", "pyarrow": "25.0.1"},
    }

    stack_path = stage / "stack_metadata_feasibility_v1.csv"
    pair_path = stage / "candidate_pair_support_v1.csv"
    primary_path = stage / "primary_eligible_pair_registry_draft_v1.csv"
    summary_path = stage / "metadata_feasibility_summary_v1.json"
    write_csv(stack_path, stack_rows)
    write_csv(pair_path, pair_rows)
    write_csv(primary_path, primary_rows, list(pair_rows[0]))
    summary["output_hashes"] = {
        path.name: sha256(path) for path in [stack_path, pair_path, primary_path]
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"counts": summary["counts"], "eligible_pair_counts_at_20_common_tasks": support_grid}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
