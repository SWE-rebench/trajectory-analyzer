#!/usr/bin/env python3
"""Extract deterministic rollout metrics from one or more trajectory JSON files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from parser import (
    CODEBOOK_VERSION,
    PARSER_ID,
    SUPPORTED_PRIMARY_FORMATS,
    compute_rollout_metrics,
    normalize_input,
    parse_trajectory,
)


def extract_one(
    path: Path,
    schema: str,
    parser_format: str | None,
    repository_root: str | None,
) -> dict[str, Any]:
    trajectory = json.loads(path.read_text(encoding="utf-8"))
    normalized = normalize_input(trajectory, schema, parser_format)
    root = repository_root or normalized.get("repository_root") or normalized.get("repo_root")
    if not root:
        raise ValueError(
            f"{path}: metrics require --repository-root or repository_root in the trajectory"
        )
    root_source = "cli" if repository_root else str(
        normalized.get("repository_root_source") or "trajectory"
    )
    parsed = parse_trajectory(
        normalized,
        rollout_id=path.stem,
        repository_root=str(root),
        repository_root_source=root_source,
    )
    return {
        "trajectory": path.as_posix(),
        "parser_id": PARSER_ID,
        "codebook_version": CODEBOOK_VERSION,
        "event_count": len(parsed["events"]),
        "diagnostics": parsed["diagnostics"],
        "metrics": compute_rollout_metrics(parsed["events"]),
    }


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("trajectories", nargs="+", type=Path)
    cli.add_argument(
        "--schema",
        choices=("auto", "messages", "evaluated-final", "normalized-events"),
        default="auto",
    )
    cli.add_argument("--parser-format", choices=tuple(sorted(SUPPORTED_PRIMARY_FORMATS)))
    cli.add_argument("--repository-root")
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()

    records = [
        extract_one(path, args.schema, args.parser_format, args.repository_root)
        for path in args.trajectories
    ]
    payload = json.dumps(records, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
