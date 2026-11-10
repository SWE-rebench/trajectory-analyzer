#!/usr/bin/env python3
"""Analyze the Python trajectory panel using the repository's parser.

This is a panel analysis, not the original 134k-corpus
pipeline. See README for the population and parser-version limitations.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import re
import shlex
import statistics
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from parser import PARSER_ID, compute_rollout_metrics, normalize_input, parse_trajectory
from python_command_methods import classify, explicit_pytest_quiet

HERE = Path(__file__).resolve().parent
DATASET = "nebius/SWE-rebench-leaderboard-trajectories"
DEFAULT_CONFIG = HERE / "config" / "python_analysis.json"
METRICS = (
    "operational_action_count", "read_share", "edit_share", "diff_inspection_share",
    "sed_print_share", "python_edit_share", "quiet_pytest_share", "native_edit_share",
    "redirect_edit_share", "native_read_share", "exploration_share",
    "non_test_repository_write_without_observed_failure_count", "formal_test_invocation_count",
    "repeated_unsuccessful_command_count", "unproductive_tail_length",
)


def safe_root(repository: str) -> str:
    """Use the roster-derived project basename, as in the historical analysis."""
    name = PurePosixPath(repository.rstrip("/")).name
    if not name or name in {".", ".."} or "/" not in repository:
        raise ValueError(f"invalid repository identifier: {repository!r}")
    return "/" + name


def rate(items: list[dict[str, Any]], predicate) -> float | None:
    return sum(bool(predicate(item)) for item in items) / len(items) if items else None


def unwrap_codex_shell(command: str) -> tuple[str, bool]:
    """Decode only a single unambiguous shell wrapper; never execute it."""
    looks_wrapped = bool(re.match(r"^\s*(?:/[^\s]+/)?(?:bash|sh|zsh)\s+-[A-Za-z]*c[A-Za-z]*\b", command, re.I))
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return command, not looks_wrapped
    if len(tokens) >= 2 and PurePosixPath(tokens[0]).name.lower() in {"bash", "sh", "zsh"} and re.fullmatch(r"-[A-Za-z]*c[A-Za-z]*", tokens[1]):
        return (tokens[2], True) if len(tokens) == 3 else (command, False)
    return command, True


def parse_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return redacted metrics; never serialize commands, observations or patches."""
    for name in ("trajectory_id", "task_id", "repository", "agent_config_id", "scaffold", "trajectory"):
        if record.get(name) is None or record[name] == "":
            raise ValueError(f"missing required field {name}")
    trajectory = json.loads(record["trajectory"]) if isinstance(record["trajectory"], str) else record["trajectory"]
    unparsed_file_changes = sum(e.get("type") == "file_change" for e in trajectory.get("events", []))
    normalized = normalize_input(trajectory, "normalized-events")
    parsed = parse_trajectory(normalized, str(record["trajectory_id"]), safe_root(record["repository"]), "repository")
    events = parsed["events"]
    metrics = compute_rollout_metrics(events)
    sem = [e for e in events if e["semantic_composition_eligible"]]
    reads = [e for e in sem if e["primary_operation"] == "READ_FILE"]
    edits = [e for e in sem if e["primary_operation"] == "MUTATE_FILE"]
    diffs = [e for e in sem if e["primary_operation"] == "INSPECT_WORKSPACE_CHANGES"]
    methods: dict[str, dict[str, Any]] = {}
    wrapper_failures = 0
    for event in events:
        command = str(event["command"])
        decoded = True
        if record["scaffold"] == "codex" and str(event["original_tool"]).lower() == "shell":
            command, decoded = unwrap_codex_shell(command)
        wrapper_failures += not decoded
        operation = event["primary_operation"]
        action = {"READ_FILE": "READ_SOURCE", "MUTATE_FILE": "EDIT_SOURCE"}.get(operation, "")
        # Legacy method labels were attached to contained test calls, not just primary tests.
        if event["contains_formal_test"] and operation != "NO_REPOSITORY_OPERATION":
            test = classify(str(event["original_tool"]), command, "RUN_FORMAL_TEST")
        else:
            test = {"test_runner": "not_test"}
        method = classify(str(event["original_tool"]), command, action)
        method["test_runner"] = test["test_runner"]
        method["quiet"] = explicit_pytest_quiet(command, test["test_runner"])
        method["decoded"] = decoded
        methods[event["event_id"]] = method
    pytests = [e for e in events if e["contains_formal_test"] and e["primary_operation"] != "NO_REPOSITORY_OPERATION" and methods[e["event_id"]]["test_runner"] in {"pytest_direct", "python_m_pytest"}]
    n = len(sem)
    metrics.update({
        "edit_share": len(edits) / n if n else None,
        "diff_inspection_share": len(diffs) / n if n else None,
        "sed_print_share": rate(reads, lambda e: methods[e["event_id"]]["read_method"] == "sed_print"),
        "python_edit_share": rate(edits, lambda e: methods[e["event_id"]]["edit_method"] in {"python_inline_mutation", "python_script_mutation"}),
        "native_edit_share": rate(edits, lambda e: methods[e["event_id"]]["edit_method"] in {"dedicated_edit_tool", "apply_patch"}),
        "redirect_edit_share": rate(edits, lambda e: methods[e["event_id"]]["edit_method"] in {"shell_redirection", "tee_write"}),
        "native_read_share": rate(reads, lambda e: methods[e["event_id"]]["read_method"] == "dedicated_read_tool"),
        "quiet_pytest_share": rate(pytests, lambda e: methods[e["event_id"]]["quiet"]),
    })
    if any(not methods[e["event_id"]]["decoded"] for e in reads):
        metrics["sed_print_share"] = metrics["native_read_share"] = None
    if any(not methods[e["event_id"]]["decoded"] for e in edits):
        for key in ("python_edit_share", "native_edit_share", "redirect_edit_share"):
            metrics[key] = None
    if any(not methods[e["event_id"]]["decoded"] and e["contains_formal_test"] for e in events):
        metrics["quiet_pytest_share"] = None
    resolved = record.get("resolved")
    if isinstance(resolved, int) and not isinstance(resolved, bool) and resolved in (0, 1):
        resolved = bool(resolved)
    elif resolved is not None and not isinstance(resolved, bool):
        raise ValueError(f"resolved must be bool, 0, 1, or null for {record['trajectory_id']}")
    return {
        "trajectory_id": str(record["trajectory_id"]), "task_id": str(record["task_id"]),
        "repository": str(record["repository"]), "agent_config_id": str(record["agent_config_id"]),
        "canonical_model": record.get("canonical_model"), "scaffold": str(record["scaffold"]),
        "reasoning_effort": record.get("reasoning_effort"), "run_index": record.get("run_index"),
        "resolved": resolved, "event_count": len(events),
        "unparsed_file_changes": unparsed_file_changes, "shell_wrapper_decode_failures": wrapper_failures,
        "metrics": {k: metrics[k] for k in METRICS},
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def public_files(revision: str, cache: Path) -> tuple[list[Path], str]:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install the Python-panel requirements in a virtual environment for Hugging Face dataset access") from exc
    info = HfApi().dataset_info(DATASET, revision=revision)
    sha = info.sha
    if not sha:
        raise RuntimeError("Could not resolve dataset revision to an immutable commit")
    directory = Path(snapshot_download(repo_id=DATASET, repo_type="dataset", revision=sha,
                                       allow_patterns="data/python_spring2026/*.parquet", cache_dir=str(cache)))
    files = sorted(directory.glob("data/python_spring2026/*.parquet"))
    if not files:
        raise RuntimeError("No Python-panel parquet files found in the dataset revision")
    return files, sha


def input_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix == ".parquet":
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if files:
            return files
    raise ValueError(f"Expected a parquet file or directory containing parquet files: {path}")


def parquet_records(files: list[Path]) -> Iterable[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Install pyarrow in a virtual environment") from exc
    required = ["trajectory_id", "agent_config_id", "canonical_model", "scaffold", "reasoning_effort", "run_index", "resolved", "trajectory"]
    for path in files:
        source = pq.ParquetFile(path)
        names = set(source.schema_arrow.names)
        missing = set(required) - names
        task_names = [x for x in ("instance_id", "task_id") if x in names]
        repo_names = [x for x in ("repo", "repository") if x in names]
        if missing or not task_names or not repo_names:
            raise ValueError(f"{path}: missing columns {sorted(missing)} or task/repository aliases")
        for batch in source.iter_batches(batch_size=32, columns=required + task_names + repo_names):
            for row in batch.to_pylist():
                if len(task_names) == 2 and row[task_names[0]] != row[task_names[1]]:
                    raise ValueError(f"{path}: task ID aliases disagree")
                if len(repo_names) == 2 and row[repo_names[0]] != row[repo_names[1]]:
                    raise ValueError(f"{path}: repository aliases disagree")
                row["task_id"] = row[task_names[0]]
                row["repository"] = row[repo_names[0]]
                yield row


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Equal-task means: average defined runs within task, then tasks within stack."""
    cells: dict[tuple[str, str, str], list[float]] = collections.defaultdict(list)
    metadata: dict[str, dict[str, Any]] = {}
    support: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for record in records:
        stack = record["agent_config_id"]
        meta = {k: record.get(k) for k in ("canonical_model", "scaffold", "reasoning_effort")}
        if stack in metadata and metadata[stack] != meta:
            raise ValueError(f"inconsistent metadata for {stack}")
        metadata[stack] = meta
        support[stack]["rollouts"] += 1
        if record.get("unparsed_file_changes", 0):
            support[stack]["excluded"] += 1
            continue
        for metric in METRICS:
            value = record["metrics"].get(metric)
            if value is not None:
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"invalid {metric} for {record['trajectory_id']}")
                cells[(stack, record["task_id"], metric)].append(float(value))
    result = []
    for stack in sorted(metadata):
        for metric in METRICS:
            matched = [(key, values) for key, values in cells.items() if key[0] == stack and key[2] == metric]
            means = [statistics.fmean(values) for _, values in matched]
            result.append({"agent_config_id": stack, **metadata[stack], "metric": metric,
                           "task_equal_mean": statistics.fmean(means) if means else None,
                           "tasks_defined": len(means), "rollouts_defined": sum(len(v) for _, v in matched),
                           "stack_rollouts": support[stack]["rollouts"],
                           "rollouts_excluded_unparsed_file_change": support[stack]["excluded"]})
    return result


def build_tables(profiles: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Project fixed paper-panel configurations while retaining metric support."""
    lookup = {(p["agent_config_id"], p["metric"]): p for p in profiles}
    if len(lookup) != len(profiles):
        raise ValueError("Duplicate stack/metric profile")
    tables = {}
    for table, selections, metrics in (
        ("generation_effort", config["generation_effort"], config["generation_effort_metrics"]),
        ("scaffold", config["scaffolds"], config["scaffold_metrics"]),
    ):
        labels = [s["label"] for s in selections]
        if len(set(labels)) != len(labels) or len(set(metrics)) != len(metrics):
            raise ValueError(f"Duplicate selection or metric in {table}")
        if set(metrics) - set(METRICS):
            raise ValueError(f"Unknown metrics in {table}")
        rows = []
        for selection in selections:
            for metric in metrics:
                profile = lookup.get((selection["agent_config_id"], metric))
                status = "missing_configuration" if profile is None else (
                    "undefined" if profile["task_equal_mean"] is None else (
                        "partial_file_changes_excluded" if profile["rollouts_excluded_unparsed_file_change"] else "available"
                    )
                )
                rows.append({"configuration": selection["label"], "agent_config_id": selection["agent_config_id"],
                             "metric": metric, "task_equal_mean": profile["task_equal_mean"] if profile else None,
                             "tasks_defined": profile["tasks_defined"] if profile else 0,
                             "rollouts_defined": profile["rollouts_defined"] if profile else 0,
                             "stack_rollouts": profile["stack_rollouts"] if profile else 0,
                             "rollouts_excluded_unparsed_file_change": profile["rollouts_excluded_unparsed_file_change"] if profile else 0,
                             "status": status})
        tables[table] = rows
    return tables


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--input", type=Path, help="local parquet file or directory; omit to download the HF Python config")
    cli.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    cli.add_argument("--revision", help="HF dataset revision; defaults to the config's pinned revision")
    cli.add_argument("--require-complete-panel", action="store_true", help="fail if trajectory, task, or configuration counts differ from the config")
    cli.add_argument("--cache", type=Path, default=HERE / ".cache" / "python")
    cli.add_argument("--output", type=Path, default=HERE / "outputs" / "python")
    cli.add_argument("--overwrite", action="store_true")
    args = cli.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["dataset"] != DATASET or config["config"] != "python_spring2026":
        raise ValueError("Unsupported Python dataset or config")
    files, revision = (input_files(args.input), None) if args.input else public_files(args.revision or config["revision"], args.cache)
    output = args.output
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output} is not empty; use --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    ids = set()
    for record in parquet_records(files):
        parsed = parse_record(record)
        if parsed["trajectory_id"] in ids:
            raise ValueError(f"duplicate trajectory ID: {parsed['trajectory_id']}")
        ids.add(parsed["trajectory_id"])
        records.append(parsed)
    if not records:
        raise ValueError("No trajectories in input")
    records.sort(key=lambda r: r["trajectory_id"])
    observed = {"trajectories": len(records), "tasks": len({r["task_id"] for r in records}),
                "configurations": len({r["agent_config_id"] for r in records})}
    if args.require_complete_panel and observed != config["expected_panel"]:
        raise ValueError(f"Incomplete Python panel: observed {observed}, expected {config['expected_panel']}")
    with (output / "rollout_metrics.jsonl").open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n")
    profiles = aggregate(records)
    write_csv(output / "task_equal_profiles.csv", profiles,
              ["agent_config_id", "canonical_model", "scaffold", "reasoning_effort", "metric", "task_equal_mean", "tasks_defined", "rollouts_defined", "stack_rollouts", "rollouts_excluded_unparsed_file_change"])
    tables = build_tables(profiles, config)
    table_fields = ["configuration", "agent_config_id", "metric", "task_equal_mean", "tasks_defined", "rollouts_defined", "stack_rollouts", "rollouts_excluded_unparsed_file_change", "status"]
    for name, table in tables.items():
        write_csv(output / (name + ".csv"), table, table_fields)
    digest = hashlib.sha256((HERE / "parser.py").read_bytes()).hexdigest()
    summary = {"dataset": DATASET if revision else "local parquet", "dataset_revision": revision,
               "input_files": len(files), "input_sha256": {p.name: sha256_file(p) for p in files},
               "parser_id": PARSER_ID, "parser_sha256": digest, "config_sha256": sha256_file(args.config),
               **observed, "expected_panel": config["expected_panel"],
               "complete_panel": observed == config["expected_panel"],
               "events": sum(r["event_count"] for r in records),
               "unparsed_file_change_events": sum(r["unparsed_file_changes"] for r in records),
               "rollouts_with_unparsed_file_changes": sum(bool(r["unparsed_file_changes"]) for r in records),
               "shell_wrapper_decode_failures": sum(r["shell_wrapper_decode_failures"] for r in records),
               "note": "110-task panel analysis; not the frozen full-corpus matched/null/retrieval estimands."}
    (output / "summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
