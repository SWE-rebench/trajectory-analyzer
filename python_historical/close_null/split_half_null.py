#!/usr/bin/env python3
"""Run the frozen split-half within-stack null control with parser-4 anchors."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Keep each worker single-threaded; parallelism is across deterministic split runs.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

STAGE = Path(__file__).resolve().parent
LOCK = STAGE.parent / "frozen/PARSER_V4_NULL_INPUTS_v1.lock.json"
RESULTS = STAGE / "null_results"
SEEDS = STAGE / "seeds"
METRICS = [
    "operational_action_count",
    "exploration_share",
    "non_test_repository_write_without_observed_failure_count",
    "formal_test_invocation_count",
    "repeated_unsuccessful_command_count",
    "unproductive_tail_length",
]
PRIMARY_METRICS = {
    "operational_action_count", "exploration_share", "formal_test_invocation_count"
}
DRAWS = 10_000
PRIMARY_SPLIT_RUNS = 100
SENSITIVITY_REPLICATES_PER_SLOT = 20
MIN_TASKS = 40
SPLIT_KEY = "split-half-null-control-v1"
BOOT_KEY = "split-half-null-control-bootstrap-v1"
MAXT_KEY = "split-half-null-control-maxT-v1"

# Read-only globals inherited by fork workers.
PANELS: list[dict[str, Any]] = []


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, data: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not data and not fieldnames:
        raise ValueError(f"No rows or schema for {path}")
    names = fieldnames or list(data[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)


def stable_seed(key: str, *parts: object) -> int:
    payload = "|".join([key, *(str(part) for part in parts)])
    return int(hashlib.sha256(payload.encode()).hexdigest()[:16], 16)


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def q(values: list[float], probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability, method="linear"))


def split_jobs() -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for index in range(PRIMARY_SPLIT_RUNS):
        jobs.append({
            "design": "primary_random_exclusion",
            "split_seed_index": index,
            "root_seed_uint64": stable_seed(SPLIT_KEY, "primary", index),
            "excluded_slot": None,
            "partition_replicate": None,
        })
    index = 0
    for excluded_slot in range(5):
        for replicate in range(SENSITIVITY_REPLICATES_PER_SLOT):
            jobs.append({
                "design": "forced_excluded_slot_sensitivity",
                "split_seed_index": index,
                "root_seed_uint64": stable_seed(SPLIT_KEY, "sensitivity", excluded_slot, replicate),
                "excluded_slot": excluded_slot,
                "partition_replicate": replicate,
            })
            index += 1
    return jobs


def load_inputs() -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    lock = json.loads(LOCK.read_text())
    if lock["status"] != "frozen_before_execution" or not lock["execution_authorized_by_user"]:
        raise RuntimeError("Null-control execution is not authorized")
    for item in lock["sources"].values():
        path = Path(item["path"])
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"Frozen source hash mismatch: {path}")
    source = {key: Path(item["path"]) for key, item in lock["sources"].items()}

    real_profiles = rows(source["real_close1_profiles"])
    complete_real = [
        r for r in real_profiles
        if r["in_close_1pp_primary"] == "true" and r["all_six_anchors_supported"] == "true"
    ]
    complete_ids = {r["candidate_pair_id"] for r in complete_real}
    if not complete_real or len(complete_real) != len(complete_ids) or len(complete_real) > 65:
        raise RuntimeError("Invalid complete real Close-1 pair set")

    task_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    selected_rollouts: set[str] = set()
    for row in rows(source["balanced_pair_task_registry"]):
        if row["candidate_pair_id"] not in complete_ids:
            continue
        task_rows[row["candidate_pair_id"]].append(row)
        selected_rollouts.update(json.loads(row["stack_a_rollout_ids_json"]))
        selected_rollouts.update(json.loads(row["stack_b_rollout_ids_json"]))
    if len(task_rows) != len(complete_real) or any(len(items) != 50 for items in task_rows.values()):
        raise RuntimeError("Expected every complete real pair to have 50 frozen tasks")

    metric_map: dict[str, list[float | None]] = {
        rollout_id: [None] * len(METRICS) for rollout_id in selected_rollouts
    }
    defined_map: dict[str, list[bool]] = {
        rollout_id: [False] * len(METRICS) for rollout_id in selected_rollouts
    }
    metric_index = {metric: i for i, metric in enumerate(METRICS)}
    columns = ["rollout_id", "metric_id", "value", "defined"]
    for batch in pq.ParquetFile(source["rollout_anchor_metrics"]).iter_batches(
        batch_size=200_000, columns=columns
    ):
        data = batch.to_pydict()
        for i in range(batch.num_rows):
            rollout_id = data["rollout_id"][i]
            if rollout_id not in metric_map:
                continue
            metric = data["metric_id"][i]
            if metric not in metric_index:
                continue
            j = metric_index[metric]
            value = data["value"][i]
            is_defined = bool(data["defined"][i]) and value is not None
            metric_map[rollout_id][j] = None if value is None else float(value)
            defined_map[rollout_id][j] = is_defined

    panels: list[dict[str, Any]] = []
    for pair_id in sorted(complete_ids):
        items = sorted(task_rows[pair_id], key=lambda r: r["task_id"])
        for arm in ("a", "b"):
            stack_id = items[0][f"stack_{arm}_id"]
            pseudo_pair_id = f"{pair_id}__stack_{arm}"
            values = np.full((50, 5, len(METRICS)), np.nan, dtype=np.float64)
            defined = np.zeros((50, 5, len(METRICS)), dtype=bool)
            rollout_ids: list[tuple[str, ...]] = []
            task_ids: list[str] = []
            for task_index, item in enumerate(items):
                ids = tuple(json.loads(item[f"stack_{arm}_rollout_ids_json"]))
                if len(ids) != 5 or len(set(ids)) != 5:
                    raise RuntimeError(f"Expected five unique rollouts: {pseudo_pair_id} {item['task_id']}")
                task_ids.append(item["task_id"])
                rollout_ids.append(ids)
                for slot, rollout_id in enumerate(ids):
                    if rollout_id not in metric_map:
                        raise RuntimeError(f"Missing rollout metrics: {rollout_id}")
                    for metric_index_value in range(len(METRICS)):
                        if defined_map[rollout_id][metric_index_value]:
                            values[task_index, slot, metric_index_value] = metric_map[rollout_id][metric_index_value]  # type: ignore[arg-type]
                            defined[task_index, slot, metric_index_value] = True
            panels.append({
                "pseudo_pair_id": pseudo_pair_id,
                "source_real_pair_id": pair_id,
                "source_arm": arm,
                "stack_id": stack_id,
                "task_ids": tuple(task_ids),
                "rollout_ids": tuple(rollout_ids),
                "values": values,
                "defined": defined,
            })
    expected_panels = 2 * len(complete_real)
    if len(panels) != expected_panels or len({p["pseudo_pair_id"] for p in panels}) != expected_panels:
        raise RuntimeError("Expected two distinct pair-arm pseudo panels per complete real pair")
    return panels, complete_real, lock


def analyze_job(job: dict[str, Any]) -> dict[str, Any]:
    assignment_hash = hashlib.sha256()
    excluded_counts = Counter()
    profile_rows: list[dict[str, Any]] = []
    anchor_rows: list[dict[str, Any]] = []

    for panel in PANELS:
        arm_a_slots = np.empty((50, 2), dtype=np.int8)
        arm_b_slots = np.empty((50, 2), dtype=np.int8)
        excluded_slots = np.empty(50, dtype=np.int8)
        for task_index in range(50):
            # Cell-keyed randomization ensures that a recurring exact stack x task
            # cell gets one identical assignment in every real-pair arm panel.
            cell_rng = np.random.default_rng(stable_seed(
                SPLIT_KEY, "cell", job["design"], job["root_seed_uint64"],
                panel["stack_id"], panel["task_ids"][task_index]
            ))
            if job["design"] == "primary_random_exclusion":
                permutation = cell_rng.permutation(5)
                arm_a, arm_b, excluded = permutation[:2], permutation[2:4], int(permutation[4])
            else:
                excluded = int(job["excluded_slot"])
                remaining = np.asarray([slot for slot in range(5) if slot != excluded], dtype=np.int8)
                permutation = cell_rng.permutation(remaining)
                arm_a, arm_b = permutation[:2], permutation[2:]
            arm_a_slots[task_index] = arm_a
            arm_b_slots[task_index] = arm_b
            excluded_slots[task_index] = excluded
            excluded_counts[excluded] += 1
            assignment_hash.update(
                f"{panel['pseudo_pair_id']}|{panel['task_ids'][task_index]}|"
                f"{int(arm_a[0])},{int(arm_a[1])}|{int(arm_b[0])},{int(arm_b[1])}|{excluded}\n".encode()
            )

        task_diffs: dict[str, np.ndarray] = {}
        nominal_flags: dict[str, bool] = {}
        observed_t: dict[str, float] = {}
        perm_t_by_metric: dict[str, np.ndarray] = {}
        max_t_p: dict[str, float] = {}
        metric_base: dict[str, dict[str, Any]] = {}

        for metric_index_value, metric in enumerate(METRICS):
            differences: list[float] = []
            for task_index in range(50):
                a_slots = arm_a_slots[task_index]
                b_slots = arm_b_slots[task_index]
                is_defined = panel["defined"][task_index, :, metric_index_value]
                if not bool(np.all(is_defined[a_slots])) or not bool(np.all(is_defined[b_slots])):
                    continue
                values = panel["values"][task_index, :, metric_index_value]
                differences.append(float(values[b_slots].mean() - values[a_slots].mean()))
            supported = len(differences) >= MIN_TASKS
            delta = ci_low = ci_high = None
            nominal = False
            if supported:
                vector = np.asarray(differences, dtype=np.float64)
                task_diffs[metric] = vector
                delta = float(vector.mean())
                boot_rng = np.random.default_rng(stable_seed(
                    BOOT_KEY, job["design"], job["split_seed_index"],
                    panel["pseudo_pair_id"], metric
                ))
                indices = boot_rng.integers(
                    0, len(vector), size=(DRAWS, len(vector)), dtype=np.int16
                )
                bootstrap = vector[indices].mean(axis=1)
                ci_low, ci_high = (
                    float(x) for x in np.quantile(
                        bootstrap, [.025, .975], method="linear"
                    )
                )
                nominal = ci_low > 0 or ci_high < 0
            nominal_flags[metric] = nominal
            metric_base[metric] = {
                "paired_defined_tasks": len(differences),
                "support_status": "supported" if supported else "unsupported",
                "delta_b_minus_a_native": delta,
                "ci95_low_native": ci_low,
                "ci95_high_native": ci_high,
                "nominal_ci_excludes_zero": bool_text(nominal),
            }

        # Same shared-task sign-flip max-|T| construction as the real-pair pipeline.
        maxt_rng = np.random.default_rng(stable_seed(
            MAXT_KEY, job["design"], job["split_seed_index"], panel["pseudo_pair_id"]
        ))
        signs = maxt_rng.choice(np.asarray([-1.0, 1.0]), size=(DRAWS, 50))
        raw_p: dict[str, float] = {}
        for metric in METRICS:
            if metric not in task_diffs:
                continue
            vector = task_diffs[metric]
            # Supported tasks are always a prefix only when no values are missing, so
            # reconstruct the exact task mask from the slot assignments.
            mask_values: list[bool] = []
            metric_index_value = METRICS.index(metric)
            for task_index in range(50):
                is_defined = panel["defined"][task_index, :, metric_index_value]
                mask_values.append(
                    bool(np.all(is_defined[arm_a_slots[task_index]]))
                    and bool(np.all(is_defined[arm_b_slots[task_index]]))
                )
            mask = np.asarray(mask_values, dtype=bool)
            if int(mask.sum()) != len(vector):
                raise RuntimeError("Task mask and difference vector disagree")
            mean = float(vector.mean())
            sd = float(vector.std(ddof=1))
            observed = (
                math.copysign(math.inf, mean) if sd == 0 and mean != 0
                else (0.0 if sd == 0 else mean / (sd / math.sqrt(len(vector))))
            )
            observed_t[metric] = observed
            means = (signs[:, mask] @ vector) / len(vector)
            sum_squares = float(np.dot(vector, vector))
            variances = np.maximum(
                (sum_squares - len(vector) * means * means) / (len(vector) - 1), 0.0
            )
            standard_errors = np.sqrt(variances / len(vector))
            perm_t = np.zeros_like(means)
            np.divide(means, standard_errors, out=perm_t, where=standard_errors > 0)
            degenerate = (standard_errors == 0) & (means != 0)
            perm_t[degenerate] = np.sign(means[degenerate]) * np.inf
            absolute_perm_t = np.abs(perm_t)
            perm_t_by_metric[metric] = absolute_perm_t
            raw_p[metric] = (
                1 + int(np.sum(absolute_perm_t >= abs(observed)))
            ) / (DRAWS + 1)
        if perm_t_by_metric:
            max_t = np.max(
                np.column_stack([perm_t_by_metric[m] for m in METRICS if m in perm_t_by_metric]),
                axis=1,
            )
            for metric in observed_t:
                max_t_p[metric] = (
                    1 + int(np.sum(max_t >= abs(observed_t[metric])))
                ) / (DRAWS + 1)

        for metric in METRICS:
            adjusted_p = max_t_p.get(metric)
            anchor_rows.append({
                "design": job["design"],
                "split_seed_index": job["split_seed_index"],
                "root_seed_uint64": job["root_seed_uint64"],
                "forced_excluded_slot": job["excluded_slot"],
                "partition_replicate": job["partition_replicate"],
                "pseudo_pair_id": panel["pseudo_pair_id"],
                "source_real_pair_id": panel["source_real_pair_id"],
                "source_arm": panel["source_arm"],
                "stack_id": panel["stack_id"],
                "metric_id": metric,
                **metric_base[metric],
                "observed_studentized_statistic": observed_t.get(metric),
                "sign_flip_p_value": raw_p.get(metric),
                "maxT_adjusted_p_value": adjusted_p,
                "maxT_adjusted_p_lt_0_05": bool_text(adjusted_p is not None and adjusted_p < .05),
            })

        supported_count = sum(metric_base[m]["support_status"] == "supported" for m in METRICS)
        complete = supported_count == 6
        nominal_count = sum(nominal_flags.values()) if complete else None
        maxt_count = (
            sum(max_t_p.get(metric, 1.0) < .05 for metric in METRICS) if complete else None
        )
        profile_rows.append({
            "design": job["design"],
            "split_seed_index": job["split_seed_index"],
            "root_seed_uint64": job["root_seed_uint64"],
            "forced_excluded_slot": job["excluded_slot"],
            "partition_replicate": job["partition_replicate"],
            "pseudo_pair_id": panel["pseudo_pair_id"],
            "source_real_pair_id": panel["source_real_pair_id"],
            "source_arm": panel["source_arm"],
            "stack_id": panel["stack_id"],
            "supported_anchor_count": supported_count,
            "all_six_anchors_supported": bool_text(complete),
            "nominal_separation_count": nominal_count,
            "maxT_separation_count": maxt_count,
        })

    return {
        "job": job,
        "assignment_sha256": assignment_hash.hexdigest(),
        "excluded_slot_counts": dict(sorted(excluded_counts.items())),
        "profiles": profile_rows,
        "anchors": anchor_rows,
    }


def summarize_thresholds(profiles: list[dict[str, Any]], design: str) -> list[dict[str, Any]]:
    complete = [
        row for row in profiles
        if row["design"] == design and row["all_six_anchors_supported"] == "true"
    ]
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in complete:
        by_seed[int(row["split_seed_index"])].append(row)
    result = []
    for threshold in (1, 2, 3):
        row: dict[str, Any] = {
            "design": design,
            "threshold": f">={threshold}",
            "complete_pseudo_pair_split_replicates": len(complete),
        }
        for method, field in (("nominal", "nominal_separation_count"), ("maxT", "maxT_separation_count")):
            count = sum(int(item[field]) >= threshold for item in complete)
            seed_fractions = [
                sum(int(item[field]) >= threshold for item in items) / len(items)
                for _, items in sorted(by_seed.items())
            ]
            row[f"{method}_count"] = count
            row[f"{method}_fraction"] = count / len(complete)
            row[f"{method}_seed_fraction_min"] = min(seed_fractions)
            row[f"{method}_seed_fraction_q25"] = q(seed_fractions, .25)
            row[f"{method}_seed_fraction_median"] = q(seed_fractions, .50)
            row[f"{method}_seed_fraction_q75"] = q(seed_fractions, .75)
            row[f"{method}_seed_fraction_max"] = max(seed_fractions)
        result.append(row)
    return result


def per_anchor_summary(anchor_rows: list[dict[str, Any]], design: str) -> list[dict[str, Any]]:
    result = []
    for metric in METRICS:
        items = [
            row for row in anchor_rows
            if row["design"] == design and row["metric_id"] == metric
            and row["support_status"] == "supported"
        ]
        by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in items:
            by_seed[int(row["split_seed_index"])].append(row)
        output: dict[str, Any] = {
            "design": design,
            "metric_id": metric,
            "is_primary_anchor": bool_text(metric in PRIMARY_METRICS),
            "supported_pseudo_pair_split_replicates": len(items),
        }
        for method, field in (
            ("nominal", "nominal_ci_excludes_zero"),
            ("maxT", "maxT_adjusted_p_lt_0_05"),
        ):
            count = sum(row[field] == "true" for row in items)
            seed_fractions = [
                sum(row[field] == "true" for row in seed_items) / len(seed_items)
                for _, seed_items in sorted(by_seed.items())
            ]
            output[f"{method}_count"] = count
            output[f"{method}_fraction"] = count / len(items)
            output[f"{method}_seed_fraction_min"] = min(seed_fractions)
            output[f"{method}_seed_fraction_median"] = q(seed_fractions, .50)
            output[f"{method}_seed_fraction_max"] = max(seed_fractions)
        result.append(output)
    return result


def distribution_rows(
    profiles: list[dict[str, Any]], real_profiles: list[dict[str, str]]
) -> list[dict[str, Any]]:
    primary = [
        row for row in profiles
        if row["design"] == "primary_random_exclusion"
        and row["all_six_anchors_supported"] == "true"
    ]
    real = [
        row for row in real_profiles
        if row["in_close_1pp_primary"] == "true"
        and row["all_six_anchors_supported"] == "true"
    ]
    output = []
    for method, null_field, real_field in (
        ("nominal", "nominal_separation_count", "anchors_with_nominal_ci_excluding_zero"),
        ("maxT", "maxT_separation_count", "anchors_with_maxT_adjusted_p_lt_0_05"),
    ):
        null_counts = Counter(int(row[null_field]) for row in primary)
        real_counts = Counter(int(row[real_field]) for row in real)
        for count in range(7):
            output.append({
                "method": method,
                "separation_count": count,
                "null_count": null_counts[count],
                "null_fraction": null_counts[count] / len(primary),
                "real_count": real_counts[count],
                "real_fraction": real_counts[count] / len(real),
            })
    return output


def exclusion_sensitivity(profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    complete = [
        row for row in profiles
        if row["design"] == "forced_excluded_slot_sensitivity"
        and row["all_six_anchors_supported"] == "true"
    ]
    output = []
    for slot in range(5):
        items = [row for row in complete if int(row["forced_excluded_slot"]) == slot]
        for threshold in (1, 2, 3):
            output.append({
                "excluded_rollout_slot": slot,
                "partition_seeds": SENSITIVITY_REPLICATES_PER_SLOT,
                "complete_pseudo_pair_split_replicates": len(items),
                "threshold": f">={threshold}",
                "nominal_count": sum(int(row["nominal_separation_count"]) >= threshold for row in items),
                "nominal_fraction": sum(int(row["nominal_separation_count"]) >= threshold for row in items) / len(items),
                "maxT_count": sum(int(row["maxT_separation_count"]) >= threshold for row in items),
                "maxT_fraction": sum(int(row["maxT_separation_count"]) >= threshold for row in items) / len(items),
                "mean_nominal_separation_count": float(np.mean([int(row["nominal_separation_count"]) for row in items])),
                "mean_maxT_separation_count": float(np.mean([int(row["maxT_separation_count"]) for row in items])),
            })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    args = parser.parse_args()

    global PANELS
    PANELS, real_profiles, lock = load_inputs()
    jobs = split_jobs()
    context = mp.get_context("fork")
    if args.workers == 1:
        run_results = [analyze_job(job) for job in jobs]
    else:
        with context.Pool(processes=args.workers) as pool:
            run_results = pool.map(analyze_job, jobs)

    profile_rows = [row for result in run_results for row in result["profiles"]]
    anchor_rows = [row for result in run_results for row in result["anchors"]]
    profile_rows.sort(key=lambda r: (r["design"], int(r["split_seed_index"]), r["pseudo_pair_id"]))
    anchor_rows.sort(key=lambda r: (r["design"], int(r["split_seed_index"]), r["pseudo_pair_id"], METRICS.index(r["metric_id"])))

    primary_seed_rows = []
    sensitivity_seed_rows = []
    for result in run_results:
        job = result["job"]
        row = {
            **job,
            "assignment_sha256": result["assignment_sha256"],
            **{f"excluded_slot_{slot}_cells": result["excluded_slot_counts"].get(slot, 0) for slot in range(5)},
        }
        if job["design"] == "primary_random_exclusion":
            primary_seed_rows.append(row)
        else:
            sensitivity_seed_rows.append(row)
    primary_seed_rows.sort(key=lambda r: int(r["split_seed_index"]))
    sensitivity_seed_rows.sort(key=lambda r: int(r["split_seed_index"]))

    threshold_rows = summarize_thresholds(profile_rows, "primary_random_exclusion")
    anchor_summary_rows = per_anchor_summary(anchor_rows, "primary_random_exclusion")
    dist_rows = distribution_rows(profile_rows, real_profiles)
    sensitivity_rows = exclusion_sensitivity(profile_rows)
    real_rows = [
        {
            "candidate_pair_id": row["candidate_pair_id"],
            "nominal_separation_count": int(row["anchors_with_nominal_ci_excluding_zero"]),
            "maxT_separation_count": int(row["anchors_with_maxT_adjusted_p_lt_0_05"]),
        }
        for row in real_profiles
        if row["in_close_1pp_primary"] == "true"
        and row["all_six_anchors_supported"] == "true"
    ]
    real_rows.sort(key=lambda r: r["candidate_pair_id"])

    RESULTS.mkdir(parents=True, exist_ok=True)
    SEEDS.mkdir(parents=True, exist_ok=True)
    write_csv(SEEDS / "primary_random_split_seeds_v1.csv", primary_seed_rows)
    write_csv(SEEDS / "excluded_rollout_sensitivity_seeds_v1.csv", sensitivity_seed_rows)
    write_csv(RESULTS / "pseudo_pair_seed_profiles_v1.csv", [r for r in profile_rows if r["design"] == "primary_random_exclusion"])
    write_csv(RESULTS / "pseudo_pair_anchor_results_v1.csv", [r for r in anchor_rows if r["design"] == "primary_random_exclusion"])
    write_csv(RESULTS / "exclusion_sensitivity_profiles_v1.csv", [r for r in profile_rows if r["design"] == "forced_excluded_slot_sensitivity"])
    write_csv(RESULTS / "exclusion_sensitivity_anchor_results_v1.csv", [r for r in anchor_rows if r["design"] == "forced_excluded_slot_sensitivity"])
    write_csv(RESULTS / "threshold_summary_v1.csv", threshold_rows)
    write_csv(RESULTS / "primary_anchor_summary_v1.csv", [r for r in anchor_summary_rows if r["is_primary_anchor"] == "true"])
    write_csv(RESULTS / "all_anchor_summary_v1.csv", anchor_summary_rows)
    write_csv(RESULTS / "false_separation_distribution_vs_real_v1.csv", dist_rows)
    write_csv(RESULTS / "real_close1_profiles_v1.csv", real_rows)
    write_csv(RESULTS / "excluded_rollout_sensitivity_v1.csv", sensitivity_rows)

    primary_complete = [
        row for row in profile_rows
        if row["design"] == "primary_random_exclusion"
        and row["all_six_anchors_supported"] == "true"
    ]
    real_nominal = [r["nominal_separation_count"] for r in real_rows]
    real_maxt = [r["maxT_separation_count"] for r in real_rows]
    null_nominal = [int(r["nominal_separation_count"]) for r in primary_complete]
    null_maxt = [int(r["maxT_separation_count"]) for r in primary_complete]
    sensitivity_ranges = {}
    for threshold in (1, 2, 3):
        selected = [r for r in sensitivity_rows if r["threshold"] == f">={threshold}"]
        sensitivity_ranges[f"at_least_{threshold}"] = {
            "nominal_fraction_min": min(float(r["nominal_fraction"]) for r in selected),
            "nominal_fraction_max": max(float(r["nominal_fraction"]) for r in selected),
            "maxT_fraction_min": min(float(r["maxT_fraction"]) for r in selected),
            "maxT_fraction_max": max(float(r["maxT_fraction"]) for r in selected),
        }

    summary = {
        "schema_version": "parser-v4-split-half-null-control-summary-v1",
        "status": "complete_pending_validation_gate",
        "method": {
            "real_complete_close1_pairs": len(real_rows),
            "pseudo_pair_panels_per_split_seed": len(PANELS),
            "primary_random_split_seeds": PRIMARY_SPLIT_RUNS,
            "forced_exclusion_partition_seeds_per_slot": SENSITIVITY_REPLICATES_PER_SLOT,
            "anchors": METRICS,
            "primary_anchors": sorted(PRIMARY_METRICS),
            "bootstrap_draws": DRAWS,
            "maxT_sign_flip_draws": DRAWS,
            "minimum_defined_rollouts_per_pseudo_arm": 2,
            "minimum_paired_defined_tasks": MIN_TASKS,
        },
        "support": {
            "primary_pseudo_pair_split_replicates_total": PRIMARY_SPLIT_RUNS * len(PANELS),
            "primary_complete_six_anchor_replicates": len(primary_complete),
            "primary_complete_fraction": len(primary_complete) / (PRIMARY_SPLIT_RUNS * len(PANELS)),
        },
        "threshold_summary": threshold_rows,
        "primary_anchor_summary": [r for r in anchor_summary_rows if r["is_primary_anchor"] == "true"],
        "false_separation_counts": {
            "null_nominal_mean": float(np.mean(null_nominal)),
            "null_nominal_median": q(null_nominal, .50),
            "null_maxT_mean": float(np.mean(null_maxt)),
            "null_maxT_median": q(null_maxt, .50),
            "real_nominal_mean": float(np.mean(real_nominal)),
            "real_nominal_median": q(real_nominal, .50),
            "real_maxT_mean": float(np.mean(real_maxt)),
            "real_maxT_median": q(real_maxt, .50),
            "null_nominal_fraction_at_or_above_real_median": sum(v >= q(real_nominal, .50) for v in null_nominal) / len(null_nominal),
            "null_maxT_fraction_at_or_above_real_median": sum(v >= q(real_maxt, .50) for v in null_maxt) / len(null_maxt),
        },
        "excluded_rollout_sensitivity_ranges": sensitivity_ranges,
        "dependence_warning": "Pseudo-pair panels and split seeds reuse stacks, tasks, and rollouts; fractions are descriptive, not IID binomial estimates.",
        "inputs_lock": {"path": str(LOCK), "sha256": sha256(LOCK)},
        "software": {"python": "3", "numpy": np.__version__, "pyarrow": pa.__version__},
        "outputs": {},
    }
    output_paths = sorted([*RESULTS.glob("*.csv"), *SEEDS.glob("*.csv")])
    for path in output_paths:
        summary["outputs"][str(path.relative_to(STAGE))] = {
            "rows": sum(1 for _ in path.open(encoding="utf-8")) - 1,
            "sha256": sha256(path),
        }
    summary_path = RESULTS / "null_control_summary_v1.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "support": summary["support"],
        "threshold_summary": threshold_rows,
        "primary_anchor_summary": summary["primary_anchor_summary"],
        "false_separation_counts": summary["false_separation_counts"],
        "excluded_rollout_sensitivity_ranges": sensitivity_ranges,
    }, indent=2))


if __name__ == "__main__":
    main()
