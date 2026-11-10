#!/usr/bin/env python3
"""Regression tests for the standalone public parser release."""
from __future__ import annotations

import unittest

from parser import compute_rollout_metrics, parse_trajectory


ROOT = "/workspace/repository"


def action(action_id: str, call_id: str, tool: str, arguments: dict) -> dict:
    return {
        "kind": "action",
        "id": action_id,
        "tool_calls": [
            {
                "id": call_id,
                "function": {"name": tool, "arguments": arguments},
            }
        ],
    }


def observation(action_id: str, call_id: str, output: str = "", exit_code: int = 0) -> dict:
    return {
        "kind": "observation",
        "id": f"observation-{call_id}",
        "parent_id": action_id,
        "tool_call_id": call_id,
        "current_dir": ROOT,
        "output": output,
        "exit_code": exit_code,
    }


class ObservationPairingRegressionTest(unittest.TestCase):
    def test_one_observation_is_not_reused_for_two_calls(self) -> None:
        trajectory = {
            "repository_root": ROOT,
            "messages": [
                {
                    "kind": "action",
                    "id": "action-multi",
                    "tool_calls": [
                        {
                            "id": "call-first",
                            "function": {
                                "name": "read",
                                "arguments": {"path": f"{ROOT}/src/first.py"},
                            },
                        },
                        {
                            "id": "call-second",
                            "function": {
                                "name": "read",
                                "arguments": {"path": f"{ROOT}/src/second.py"},
                            },
                        },
                    ],
                },
                observation("action-multi", "call-first", "first file"),
            ],
        }

        parsed = parse_trajectory(trajectory, repository_root=ROOT)

        self.assertEqual([event["pairing_status"] for event in parsed["events"]], ["paired", "missing"])
        self.assertEqual(
            [event["execution_status"] for event in parsed["events"]],
            ["executed", "observation_missing"],
        )

    def test_parent_fallback_remains_available_for_single_call_action(self) -> None:
        trajectory = {
            "repository_root": ROOT,
            "messages": [
                action("action-single", "call-single", "read", {"path": f"{ROOT}/src/file.py"}),
                {
                    "kind": "observation",
                    "id": "observation-parent-only",
                    "parent_id": "action-single",
                    "current_dir": ROOT,
                    "output": "file",
                    "exit_code": 0,
                },
            ],
        }

        parsed = parse_trajectory(trajectory, repository_root=ROOT)

        self.assertEqual(parsed["events"][0]["pairing_status"], "paired")


class ProfileMetricRegressionTest(unittest.TestCase):
    def test_added_operation_shares_use_semantic_denominator(self) -> None:
        messages = [
            action("action-search", "call-search", "bash", {"command": "rg needle src"}),
            observation("action-search", "call-search"),
            action("action-edit", "call-edit", "edit", {"file_path": f"{ROOT}/src/file.py"}),
            observation("action-edit", "call-edit"),
            action("action-build", "call-build", "bash", {"command": "pip install -e ."}),
            observation("action-build", "call-build"),
            action("action-other", "call-other", "bash", {"command": "echo done"}),
            observation("action-other", "call-other"),
        ]
        parsed = parse_trajectory({"repository_root": ROOT, "messages": messages}, repository_root=ROOT)
        metrics = compute_rollout_metrics(parsed["events"])

        self.assertEqual(metrics["semantic_denominator"], 4)
        self.assertEqual(metrics["search_share"], 0.25)
        self.assertEqual(metrics["file_mutation_share"], 0.25)
        self.assertEqual(metrics["build_package_share"], 0.25)
        self.assertEqual(metrics["other_operation_share"], 0.25)


if __name__ == "__main__":
    unittest.main()
