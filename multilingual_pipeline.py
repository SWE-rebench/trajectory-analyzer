#!/usr/bin/env python3
"""Reproduce multilingual event summaries, profiles, and retrieval tests."""
from __future__ import annotations

import argparse
import collections
import contextlib
import csv
import gzip
import hashlib
import io
import itertools
import json
import math
import os
import random
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, TextIO

from parser import compute_rollout_metrics, normalize_input, parse_trajectory

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config" / "multilingual_analysis.json"
DEFAULT_OUTPUT = HERE / "outputs" / "multilingual"
DEFAULT_CACHE = HERE / ".cache" / "multilingual"

SHELL_TOOLS = {"shell", "bash", "bash_terminal", "terminal", "command", "execute", "computer"}
PROCESS_FEATURES = ["normalized_general_action_entropy", "general_action_switch_rate"]

EVENT_FIELDS = [
    "trajectory_id", "participant", "task_id", "language", "run", "event_id", "step_index",
    "tool_call_index", "original_tool", "command_sha256", "signature_sha256", "execution_status",
    "pairing_status", "primary_operation", "contained_operations", "matched_rule_ids",
    "contains_formal_test", "formal_test_invocation_count", "contains_multiple_formal_test_invocations",
    "contains_mutation", "contains_non_test_repository_path", "contains_diff_inspection", "target_role",
    "target_role_rule_ids", "mutation_kind", "read_mode", "read_scope", "affected_paths",
    "mutation_paths", "external_redirect", "exit_code", "command_succeeded_explicit",
    "no_observed_failure", "write_success_evidence", "formal_test_result", "formal_test_result_rule",
    "formal_test_evidence", "formal_test_result_conflict", "semantic_composition_eligible",
    "attempted_action_eligible",
]
PROFILE_FIELDS = [
    "participant", "task_id", "language", "run", "trajectory_id", "scaffold", "feature_id", "value",
    "defined",
]


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    actions = list(config["general_actions"])
    if len(actions) != 10 or len(set(actions)) != 10:
        raise ValueError("general_actions must contain ten unique labels")
    if len(config["participants"]) != config["expected"]["participants"]:
        raise ValueError("participant inventory does not match expected support")
    common = set(config["common_scaffold_participants"])
    if not common < set(config["participants"]):
        raise ValueError("common-scaffold inventory must be a strict participant subset")
    return config


def feature_ids(config: dict[str, Any]) -> list[str]:
    return PROCESS_FEATURES + [f"action_share__{label}" for label in config["general_actions"]]


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else float("nan")


def fetch_bytes(url: str, attempts: int = 4) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "multilingual-reproduction"})
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as error:
            last_error = error
            if attempt + 1 == attempts:
                break
    raise RuntimeError(f"could not retrieve {url}") from last_error


def fetch_index(config: dict[str, Any]) -> list[dict[str, Any]]:
    payload = fetch_bytes(config["base_url"].rstrip("/") + "/index.jsonl")
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    participants = set(config["participants"])
    selected = [row for row in rows if row["participant_key"] in participants]
    selected.sort(key=lambda row: (config["participants"].index(row["participant_key"]), int(row["run"])))
    expected = config["expected"]
    if len(rows) != expected["shards"] + 5:
        raise RuntimeError(f"unexpected public shard count: {len(rows)}")
    if len(selected) != expected["shards"]:
        raise RuntimeError(f"unexpected selected shard count: {len(selected)}")
    if sum(int(row["rows"]) for row in selected) != expected["trajectories"]:
        raise RuntimeError("selected trajectory count does not match configuration")
    return selected


def shard_path(cache: Path, row: dict[str, Any]) -> Path:
    return cache / PurePosixPath(row["path"])


def acquire_shard(config: dict[str, Any], cache: Path, row: dict[str, Any]) -> Path:
    destination = shard_path(cache, row)
    expected_bytes = int(row["gzip_bytes"])
    if destination.exists() and destination.stat().st_size == expected_bytes:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    url = config["base_url"].rstrip("/") + "/" + row["path"]
    request = urllib.request.Request(url, headers={"User-Agent": "multilingual-reproduction"})
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=180) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target, length=1 << 20)
            if temporary.stat().st_size != expected_bytes:
                raise RuntimeError(
                    f"downloaded size mismatch for {row['path']}: {temporary.stat().st_size} != {expected_bytes}"
                )
            os.replace(temporary, destination)
            return destination
        except (urllib.error.URLError, TimeoutError, RuntimeError) as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt + 1 == 4:
                break
    raise RuntimeError(f"could not download {row['path']}") from last_error


@contextlib.contextmanager
def deterministic_gzip_text(path: Path) -> Iterator[TextIO]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                yield text


def mutation_action(event: dict[str, Any]) -> str:
    kind = event["mutation_kind"]
    if kind == "CREATE":
        return "CREATE_SOURCE"
    if kind == "DELETE":
        return "DELETE_SOURCE"
    if kind == "EDIT":
        return "EDIT_SOURCE"
    for rule in event["matched_rule_ids"]:
        if rule == "create_command":
            return "CREATE_SOURCE"
        if rule == "delete_command":
            return "DELETE_SOURCE"
        if rule in {
            "patch_or_edit", "inplace_edit", "inline_python_path_mutation",
            "inline_python_variable_path_mutation", "inline_python_path_open_write",
            "inline_python_mutation_api", "repository_redirect",
        }:
            return "EDIT_SOURCE"
    return "EDIT_SOURCE"


def general_action(event: dict[str, Any]) -> str | None:
    if not event["semantic_composition_eligible"] or event["contains_formal_test"]:
        return None
    operation = event["primary_operation"]
    if operation == "EXPLORE_REPOSITORY":
        return "EXPLORE_REPOSITORY"
    if operation == "SEARCH_REPOSITORY":
        return "SEARCH_CODE"
    if operation == "READ_FILE":
        return "READ_FILE"
    if operation == "MUTATE_FILE":
        return mutation_action(event)
    if operation == "INSPECT_WORKSPACE_CHANGES":
        return "INSPECT_DIFF"
    if operation == "PACKAGE_OR_BUILD":
        return "PACKAGE_OR_BUILD"
    if operation == "OTHER_OPERATION":
        tool = str(event["original_tool"] or "").lower().replace("-", "_")
        is_shell = tool in SHELL_TOOLS or "shell" in tool or "terminal" in tool
        return "RUN_OTHER_SHELL" if is_shell else "OTHER_TOOL"
    return None


def rollout_profile(events: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, float | None]:
    actions = config["general_actions"]
    sequence = [label for event in events if (label := general_action(event)) is not None]
    counts = collections.Counter(sequence)
    if sequence:
        entropy = -sum(
            (count / len(sequence)) * math.log(count / len(sequence))
            for count in counts.values()
        ) / math.log(len(actions))
    else:
        entropy = None
    switch_rate = None if len(sequence) < 2 else sum(
        left != right for left, right in zip(sequence, sequence[1:])
    ) / (len(sequence) - 1)
    values: dict[str, float | None] = {
        "normalized_general_action_entropy": entropy,
        "general_action_switch_rate": switch_rate,
    }
    values.update({
        f"action_share__{label}": counts[label] / len(sequence) if sequence else None
        for label in actions
    })
    return values


def repository_root(record: dict[str, Any]) -> str:
    repository = str(record.get("repository") or "").strip().rstrip("/")
    if not repository or "/" not in repository:
        raise ValueError(f"invalid repository field for {record.get('trajectory_id')}")
    return "/" + repository.rsplit("/", 1)[-1]


def event_projection(
    record: dict[str, Any], event: dict[str, Any], participant: str, run: int,
) -> dict[str, Any]:
    row = {
        "trajectory_id": record["trajectory_id"],
        "participant": participant,
        "task_id": record["instance_id"],
        "language": record["language"],
        "run": run,
        "event_id": event["event_id"],
        "step_index": event["step_index"],
        "tool_call_index": event["tool_call_index"],
        "original_tool": event["original_tool"],
        "command_sha256": hashlib.sha256(event["command"].encode("utf-8")).hexdigest(),
        "signature_sha256": hashlib.sha256(event["canonical_command_signature"].encode("utf-8")).hexdigest(),
        "contained_operations": event["contained_operations_ordered"],
    }
    for field in EVENT_FIELDS:
        if field not in row:
            row[field] = event.get(field)
    return row


def parse_public_record(record: dict[str, Any], config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, float | None]]:
    participant = str(record["participant"]["key"])
    run = int(record["run"]["index"])
    normalized = normalize_input(record, schema="normalized-events")
    parsed = parse_trajectory(
        normalized,
        rollout_id=str(record["trajectory_id"]),
        repository_root=repository_root(record),
        repository_root_source="repository_field",
    )
    metrics = compute_rollout_metrics(parsed["events"])
    return parsed["events"], {"metrics": metrics, "diagnostics": parsed["diagnostics"]}, rollout_profile(parsed["events"], config)


def extract(config: dict[str, Any], output: Path, cache: Path, overwrite: bool) -> dict[str, Any]:
    outputs = {
        "events": output / "annotated_events.jsonl.gz",
        "metrics": output / "rollout_metrics.jsonl.gz",
        "profiles": output / "rollout_profiles.csv.gz",
        "summary": output / "extraction_summary.json",
    }
    if not overwrite and any(path.exists() for path in outputs.values()):
        raise RuntimeError(f"output already exists: {output}; use --overwrite to replace it")
    output.mkdir(parents=True, exist_ok=True)
    rows = fetch_index(config)
    event_count = 0
    trajectory_count = 0
    tasks: set[str] = set()
    participants: set[str] = set()
    languages: collections.Counter[str] = collections.Counter()
    cells: collections.Counter[tuple[str, str]] = collections.Counter()
    source_event_count = 0
    compressed_bytes = 0
    failures: list[dict[str, str]] = []
    temporary = {key: path.with_suffix(path.suffix + ".tmp") for key, path in outputs.items() if key != "summary"}
    try:
        with deterministic_gzip_text(temporary["events"]) as event_stream, \
             deterministic_gzip_text(temporary["metrics"]) as metric_stream, \
             deterministic_gzip_text(temporary["profiles"]) as profile_stream:
            profile_writer = csv.DictWriter(profile_stream, fieldnames=PROFILE_FIELDS, lineterminator="\n")
            profile_writer.writeheader()
            for shard_number, shard in enumerate(rows, 1):
                path = acquire_shard(config, cache, shard)
                compressed_bytes += path.stat().st_size
                source_event_count += int(shard["events"])
                content_digest = hashlib.sha256()
                shard_rows = 0
                with gzip.open(path, "rb") as source:
                    for raw_line in source:
                        content_digest.update(raw_line)
                        if not raw_line.strip():
                            continue
                        shard_rows += 1
                        record = json.loads(raw_line)
                        participant = str(record["participant"]["key"])
                        run = int(record["run"]["index"])
                        try:
                            if participant != shard["participant_key"] or run != int(shard["run"]):
                                raise ValueError("record does not match shard identity")
                            events, metric_bundle, profile = parse_public_record(record, config)
                        except Exception as error:
                            failures.append({
                                "trajectory_id": str(record.get("trajectory_id") or ""),
                                "error": f"{type(error).__name__}: {error}",
                            })
                            continue
                        trajectory_count += 1
                        event_count += len(events)
                        task_id = str(record["instance_id"])
                        language = str(record["language"])
                        tasks.add(task_id)
                        participants.add(participant)
                        languages[language] += 1
                        cells[(participant, task_id)] += 1
                        for event in events:
                            event_stream.write(stable_json(event_projection(record, event, participant, run)) + "\n")
                        metric_stream.write(stable_json({
                            "trajectory_id": record["trajectory_id"],
                            "participant": participant,
                            "task_id": task_id,
                            "language": language,
                            "run": run,
                            "scaffold": record["participant"]["scaffold"],
                            **metric_bundle,
                        }) + "\n")
                        for feature in feature_ids(config):
                            value = profile[feature]
                            profile_writer.writerow({
                                "participant": participant,
                                "task_id": task_id,
                                "language": language,
                                "run": run,
                                "trajectory_id": record["trajectory_id"],
                                "scaffold": record["participant"]["scaffold"],
                                "feature_id": feature,
                                "value": "" if value is None else repr(value),
                                "defined": str(value is not None).lower(),
                            })
                if shard_rows != int(shard["rows"]):
                    raise RuntimeError(f"row-count mismatch for {shard['path']}")
                if content_digest.hexdigest() != shard["content_sha256"]:
                    raise RuntimeError(f"content checksum mismatch for {shard['path']}")
                print(
                    f"processed shard {shard_number}/{len(rows)}: trajectories={trajectory_count}, events={event_count}",
                    file=sys.stderr,
                )
        expected = config["expected"]
        checks = {
            "selected_shards": len(rows) == expected["shards"],
            "trajectories": trajectory_count == expected["trajectories"],
            "classified_events": event_count == expected["classified_events"],
            "tasks": len(tasks) == expected["tasks"],
            "participants": participants == set(config["participants"]),
            "languages": set(languages) == set(config["languages"]),
            "five_repetitions": len(cells) == expected["tasks"] * expected["participants"] and set(cells.values()) == {5},
            "no_failures": not failures,
        }
        if not all(checks.values()):
            raise RuntimeError(f"extraction validation failed: {checks}; failures={failures[:3]}")
        for key in ("events", "metrics", "profiles"):
            os.replace(temporary[key], outputs[key])
        summary = {
            "dataset": config["dataset"],
            "participants": len(participants),
            "tasks": len(tasks),
            "languages": dict(sorted(languages.items())),
            "trajectories": trajectory_count,
            "source_events": source_event_count,
            "classified_events": event_count,
            "downloaded_compressed_bytes": compressed_bytes,
            "outcomes_accessed": False,
            "raw_commands_saved": False,
            "raw_observations_saved": False,
            "checks": checks,
            "outputs": {
                path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in (outputs["events"], outputs["metrics"], outputs["profiles"])
            },
        }
        outputs["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return summary
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)


def read_profiles(path: Path) -> dict[tuple[str, str, str, str], list[float]]:
    values: dict[tuple[str, str, str, str], list[float]] = collections.defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["defined"] == "true":
                values[(row["language"], row["task_id"], row["participant"], row["feature_id"])].append(float(row["value"]))
    return dict(values)


def build_language_profiles(
    centered: dict[tuple[str, str, str, str], float],
    tasks_by_language: dict[str, list[str]],
    participants: list[str],
    features: list[str],
    sampled: dict[str, list[str]] | None = None,
) -> tuple[dict[tuple[str, str, str], float], dict[tuple[str, str], float]]:
    profiles: dict[tuple[str, str, str], float] = {}
    rms: dict[tuple[str, str], float] = {}
    for language in tasks_by_language:
        tasks = (sampled or tasks_by_language)[language]
        for feature in features:
            all_values = [centered[(language, task, participant, feature)] for task in tasks for participant in participants]
            rms[(language, feature)] = math.sqrt(sum(value * value for value in all_values) / len(all_values))
            for participant in participants:
                selected = [centered[(language, task, participant, feature)] for task in tasks]
                profiles[(language, participant, feature)] = sum(selected) / len(selected)
    return profiles, rms


def pool_profiles(
    profiles: dict[tuple[str, str, str], float],
    rms: dict[tuple[str, str], float],
    languages: tuple[str, ...],
    participants: list[str],
    features: list[str],
) -> tuple[dict[tuple[str, str], float], dict[str, float]]:
    pooled = {
        (participant, feature): sum(profiles[(language, participant, feature)] for language in languages) / len(languages)
        for participant in participants for feature in features
    }
    scales = {
        feature: math.sqrt(sum(rms[(language, feature)] ** 2 for language in languages) / len(languages))
        for feature in features
    }
    return pooled, scales


def retrieve(
    profiles: dict[tuple[str, str, str], float],
    rms: dict[tuple[str, str], float],
    query_languages: tuple[str, ...],
    candidate_languages: tuple[str, ...],
    participants: list[str],
    features: list[str],
    scenario: str,
    partition: str,
    direction: str,
    save_matrix: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    query_profiles, scales = pool_profiles(profiles, rms, query_languages, participants, features)
    candidate_profiles, _ = pool_profiles(profiles, rms, candidate_languages, participants, features)
    similarities: dict[tuple[str, str], float] = {}
    matrix: list[dict[str, Any]] = []
    for query in participants:
        query_vector = [query_profiles[(query, feature)] / scales[feature] for feature in features]
        for candidate in participants:
            candidate_vector = [candidate_profiles[(candidate, feature)] / scales[feature] for feature in features]
            similarity = cosine(query_vector, candidate_vector)
            similarities[(query, candidate)] = similarity
            if save_matrix:
                matrix.append({
                    "scenario": scenario,
                    "partition": partition,
                    "direction": direction,
                    "query_languages": "+".join(query_languages),
                    "candidate_languages": "+".join(candidate_languages),
                    "query_participant": query,
                    "candidate_participant": candidate,
                    "cosine_similarity": similarity,
                    "features_used": len(features),
                })
    matches: list[dict[str, Any]] = []
    for query in participants:
        ordered = sorted(participants, key=lambda candidate: (-similarities[(query, candidate)], candidate))
        rank = ordered.index(query) + 1
        self_similarity = similarities[(query, query)]
        best_nonself = max(similarities[(query, candidate)] for candidate in participants if candidate != query)
        matches.append({
            "scenario": scenario,
            "partition": partition,
            "direction": direction,
            "query_languages": "+".join(query_languages),
            "candidate_languages": "+".join(candidate_languages),
            "query_participant": query,
            "retrieved_participant": ordered[0],
            "retrieved_index": participants.index(ordered[0]),
            "self_rank": rank,
            "self_similarity": self_similarity,
            "best_nonself_similarity": best_nonself,
            "self_margin": self_similarity - best_nonself,
            "correct_top1": rank == 1,
            "correct_top3": rank <= 3,
        })
    return matrix, matches


def summarize_matches(matches: list[dict[str, Any]]) -> dict[str, float | int]:
    count = len(matches)
    return {
        "queries": count,
        "top1": sum(row["correct_top1"] for row in matches) / count,
        "top3": sum(row["correct_top3"] for row in matches) / count,
        "mean_rank": sum(row["self_rank"] for row in matches) / count,
        "mean_self_similarity": sum(row["self_similarity"] for row in matches) / count,
        "mean_best_nonself_similarity": sum(row["best_nonself_similarity"] for row in matches) / count,
        "mean_margin": sum(row["self_margin"] for row in matches) / count,
    }


def primary_evaluation(
    profiles: dict[tuple[str, str, str], float],
    rms: dict[tuple[str, str], float],
    languages: list[str],
    participants: list[str],
    features: list[str],
    scenario: str,
    save_matrix: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, float | int]]:
    matrices: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []
    directions: list[dict[str, Any]] = []
    for held_out in itertools.combinations(languages, 2):
        training = tuple(language for language in languages if language not in held_out)
        partition = "train_" + "-".join(training) + "__test_" + "-".join(held_out)
        for query, candidate, direction in (
            (training, held_out, "three_to_two"),
            (held_out, training, "two_to_three"),
        ):
            matrix, result = retrieve(
                profiles, rms, query, candidate, participants, features,
                scenario, partition, direction, save_matrix,
            )
            matrices.extend(matrix)
            matches.extend(result)
            directions.append({
                "scenario": scenario,
                "partition": partition,
                "direction": direction,
                "query_languages": "+".join(query),
                "candidate_languages": "+".join(candidate),
                **summarize_matches(result),
            })
    return matrices, matches, directions, summarize_matches(matches)


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str], compressed: bool = False) -> None:
    if compressed:
        manager = deterministic_gzip_text(path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        manager = path.open("w", encoding="utf-8", newline="")
    with manager as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def analyze(config: dict[str, Any], output: Path, overwrite: bool) -> dict[str, Any]:
    source = output / "rollout_profiles.csv.gz"
    if not source.exists():
        raise RuntimeError(f"missing extraction output: {source}")
    targets = {
        "task_features": output / "task_configuration_features.csv.gz",
        "language_profiles": output / "configuration_language_profiles.csv",
        "matrix": output / "similarity_matrix.csv.gz",
        "matches": output / "retrieval_matches.csv",
        "directions": output / "partition_direction_summary.csv",
        "scenarios": output / "scenario_summary.csv",
        "secondary": output / "secondary_summary.csv",
        "bootstrap": output / "bootstrap.csv.gz",
        "permutation": output / "permutation.csv.gz",
        "summary": output / "analysis_summary.json",
    }
    if not overwrite and any(path.exists() for path in targets.values()):
        raise RuntimeError(f"analysis output already exists: {output}; use --overwrite to replace it")
    raw = read_profiles(source)
    participants = list(config["participants"])
    languages = list(config["languages"])
    features = feature_ids(config)
    tasks_by_language: dict[str, list[str]] = {}
    for language in languages:
        tasks_by_language[language] = sorted({key[1] for key in raw if key[0] == language})
    if {language: len(tasks) for language, tasks in tasks_by_language.items()} != config["expected"]["tasks_per_language"]:
        raise RuntimeError("task-language support does not match configuration")
    cells: dict[tuple[str, str, str, str], float] = {}
    for key, values in raw.items():
        if len(values) == 5:
            cells[key] = sum(values) / len(values)
    expected_cells = config["expected"]["participants"] * config["expected"]["tasks"] * len(features)
    if len(cells) != expected_cells:
        raise RuntimeError(f"task-participant-feature support mismatch: {len(cells)} != {expected_cells}")
    centered: dict[tuple[str, str, str, str], float] = {}
    task_rows: list[dict[str, Any]] = []
    for language in languages:
        for task in tasks_by_language[language]:
            for feature in features:
                task_mean = sum(cells[(language, task, participant, feature)] for participant in participants) / len(participants)
                for participant in participants:
                    value = cells[(language, task, participant, feature)]
                    centered_value = value - task_mean
                    centered[(language, task, participant, feature)] = centered_value
                    task_rows.append({
                        "language": language,
                        "task_id": task,
                        "participant": participant,
                        "feature_id": feature,
                        "task_participant_mean": value,
                        "task_centered_value": centered_value,
                        "defined_rollouts": 5,
                    })
    profiles, rms = build_language_profiles(centered, tasks_by_language, participants, features)
    language_rows = [{
        "language": language,
        "participant": participant,
        "feature_id": feature,
        "task_adjusted_profile": profiles[(language, participant, feature)],
        "query_scale_component_rms": rms[(language, feature)],
        "tasks": len(tasks_by_language[language]),
    } for language in languages for participant in participants for feature in features]
    matrix, matches, directions, primary = primary_evaluation(
        profiles, rms, languages, participants, features, "full12", True,
    )
    scenarios: list[dict[str, Any]] = [{
        "scenario": "full12", "participants": len(participants), "features": len(features),
        **primary, "chance_top1": 1 / len(participants),
    }]
    for name, selected_features in (("composition10", features[2:]), ("process2", features[:2])):
        _, _, _, result = primary_evaluation(
            profiles, rms, languages, participants, selected_features, name, False,
        )
        scenarios.append({
            "scenario": name, "participants": len(participants), "features": len(selected_features),
            **result, "chance_top1": 1 / len(participants),
        })
    common = [participant for participant in participants if participant in config["common_scaffold_participants"]]
    _, _, _, common_result = primary_evaluation(
        profiles, rms, languages, common, features, "common_scaffold_full12", False,
    )
    scenarios.append({
        "scenario": "common_scaffold_full12", "participants": len(common), "features": len(features),
        **common_result, "chance_top1": 1 / len(common),
    })
    secondary: list[dict[str, Any]] = []
    for held_out in languages:
        training = tuple(language for language in languages if language != held_out)
        _, result = retrieve(
            profiles, rms, training, (held_out,), participants, features,
            "leave_one_out", held_out, "four_to_one", False,
        )
        secondary.append({
            "design": "leave_one_out", "query_languages": "+".join(training),
            "candidate_languages": held_out, **summarize_matches(result),
        })
    for query in languages:
        for candidate in languages:
            if query == candidate:
                continue
            _, result = retrieve(
                profiles, rms, (query,), (candidate,), participants, features,
                "ordered_pair", query + "_to_" + candidate, "one_to_one", False,
            )
            secondary.append({
                "design": "ordered_pair", "query_languages": query,
                "candidate_languages": candidate, **summarize_matches(result),
            })
    bootstrap_rng = random.Random(int(config["bootstrap_seed"]))
    bootstrap_rows: list[dict[str, Any]] = []
    for draw in range(int(config["bootstrap_draws"])):
        sampled = {
            language: [bootstrap_rng.choice(tasks_by_language[language]) for _ in tasks_by_language[language]]
            for language in languages
        }
        draw_profiles, draw_rms = build_language_profiles(
            centered, tasks_by_language, participants, features, sampled,
        )
        _, _, _, result = primary_evaluation(
            draw_profiles, draw_rms, languages, participants, features, "bootstrap", False,
        )
        bootstrap_rows.append({"draw": draw, **result})
    permutation_rng = random.Random(int(config["permutation_seed"]))
    permutation_rows: list[dict[str, Any]] = []
    exceedances = 0
    for draw in range(int(config["permutation_draws"])):
        labels = list(participants)
        permutation_rng.shuffle(labels)
        correct = sum(labels[row["retrieved_index"]] == row["query_participant"] for row in matches)
        top1 = correct / len(matches)
        exceedances += top1 >= float(primary["top1"]) - 1e-15
        permutation_rows.append({"draw": draw, "aggregate_top1": top1})
    bootstrap_top1 = [float(row["top1"]) for row in bootstrap_rows]
    leave_one_out = [row for row in secondary if row["design"] == "leave_one_out"]
    ordered_pairs = [row for row in secondary if row["design"] == "ordered_pair"]
    summary = {
        "participants": len(participants),
        "languages": languages,
        "tasks_per_language": {language: len(tasks) for language, tasks in tasks_by_language.items()},
        "features": features,
        "outcomes_accessed": False,
        "length_or_specific_tool_identity_used": False,
        "support": {
            "rollout_feature_rows": sum(len(values) for values in raw.values()),
            "task_participant_feature_cells": len(cells),
            "language_participant_profiles": len(language_rows),
        },
        "primary": {
            **primary,
            "correct_top1": sum(row["correct_top1"] for row in matches),
            "correct_top3": sum(row["correct_top3"] for row in matches),
            "chance_top1": 1 / len(participants),
            "bootstrap_ci95_top1": [percentile(bootstrap_top1, 0.025), percentile(bootstrap_top1, 0.975)],
            "bootstrap_draws": len(bootstrap_rows),
            "permutation_pvalue_top1": (exceedances + 1) / (len(permutation_rows) + 1),
            "partition_direction_top1_min": min(float(row["top1"]) for row in directions),
            "partition_direction_top1_max": max(float(row["top1"]) for row in directions),
        },
        "scenarios": scenarios[1:],
        "leave_one_out": {
            "mean_top1": sum(float(row["top1"]) for row in leave_one_out) / len(leave_one_out),
            "minimum_top1": min(float(row["top1"]) for row in leave_one_out),
            "maximum_top1": max(float(row["top1"]) for row in leave_one_out),
            "weakest_language": min(leave_one_out, key=lambda row: float(row["top1"]))["candidate_languages"],
        },
        "ordered_pairs": {
            "mean_top1": sum(float(row["top1"]) for row in ordered_pairs) / len(ordered_pairs),
            "minimum_top1": min(float(row["top1"]) for row in ordered_pairs),
            "maximum_top1": max(float(row["top1"]) for row in ordered_pairs),
        },
    }
    write_csv(targets["task_features"], task_rows, list(task_rows[0]), compressed=True)
    write_csv(targets["language_profiles"], language_rows, list(language_rows[0]))
    write_csv(targets["matrix"], matrix, list(matrix[0]), compressed=True)
    write_csv(targets["matches"], matches, list(matches[0]))
    write_csv(targets["directions"], directions, list(directions[0]))
    write_csv(targets["scenarios"], scenarios, list(scenarios[0]))
    write_csv(targets["secondary"], secondary, list(secondary[0]))
    write_csv(targets["bootstrap"], bootstrap_rows, list(bootstrap_rows[0]), compressed=True)
    write_csv(targets["permutation"], permutation_rows, list(permutation_rows[0]), compressed=True)
    targets["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_cli() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("command", choices=("extract", "analyze", "all"))
    cli.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    cli.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    cli.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    cli.add_argument("--overwrite", action="store_true")
    return cli


def main() -> None:
    args = build_cli().parse_args()
    config = load_config(args.config)
    if args.command in {"extract", "all"}:
        summary = extract(config, args.output, args.cache, args.overwrite)
        print(json.dumps(summary, indent=2, sort_keys=True))
    if args.command in {"analyze", "all"}:
        summary = analyze(config, args.output, args.overwrite)
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
