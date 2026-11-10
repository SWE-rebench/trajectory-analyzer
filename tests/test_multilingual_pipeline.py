#!/usr/bin/env python3
"""Tests for the public multilingual reproduction pipeline."""
from __future__ import annotations

import math
import unittest
from pathlib import Path

from multilingual_pipeline import (
    event_projection,
    feature_ids,
    general_action,
    load_config,
    parse_public_record,
    rollout_profile,
)


HERE = Path(__file__).resolve().parents[1]
CONFIG = HERE / "config" / "multilingual_analysis.json"


def event(operation: str, **overrides):
    value = {
        "semantic_composition_eligible": True,
        "contains_formal_test": False,
        "primary_operation": operation,
        "mutation_kind": "NOT_APPLICABLE",
        "matched_rule_ids": [],
        "original_tool": "Bash",
    }
    value.update(overrides)
    return value


class MultilingualFeatureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(CONFIG)

    def test_expected_support_and_feature_inventory(self) -> None:
        self.assertEqual(len(self.config["participants"]), 16)
        self.assertEqual(len(self.config["common_scaffold_participants"]), 13)
        self.assertEqual(len(self.config["languages"]), 5)
        self.assertEqual(len(feature_ids(self.config)), 12)

    def test_general_action_projection(self) -> None:
        self.assertEqual(general_action(event("SEARCH_REPOSITORY")), "SEARCH_CODE")
        self.assertEqual(
            general_action(event("MUTATE_FILE", mutation_kind="EDIT")),
            "EDIT_SOURCE",
        )
        self.assertEqual(general_action(event("OTHER_OPERATION")), "RUN_OTHER_SHELL")
        self.assertIsNone(general_action(event("RUN_FORMAL_TEST", contains_formal_test=True)))

    def test_rollout_profile_uses_ten_category_entropy_scale(self) -> None:
        events = [
            event("EXPLORE_REPOSITORY"),
            event("SEARCH_REPOSITORY"),
            event("MUTATE_FILE", mutation_kind="EDIT"),
            event("RUN_FORMAL_TEST", contains_formal_test=True),
        ]
        profile = rollout_profile(events, self.config)
        expected_entropy = math.log(3) / math.log(10)
        self.assertAlmostEqual(profile["normalized_general_action_entropy"], expected_entropy)
        self.assertEqual(profile["general_action_switch_rate"], 1.0)
        self.assertEqual(profile["action_share__EXPLORE_REPOSITORY"], 1 / 3)
        self.assertEqual(profile["action_share__SEARCH_CODE"], 1 / 3)
        self.assertEqual(profile["action_share__EDIT_SOURCE"], 1 / 3)
        self.assertEqual(profile["action_share__RUN_OTHER_SHELL"], 0.0)


class PublicRecordTest(unittest.TestCase):
    def test_normalized_record_is_classified_without_local_metadata(self) -> None:
        config = load_config(CONFIG)
        record = {
            "schema_version": "1.1.0",
            "trajectory_id": "sample__run-0__owner__repo-1",
            "instance_id": "owner__repo-1",
            "language": "python",
            "repository": "owner/repo",
            "participant": {
                "key": "fable-5",
                "type": "common_scaffold_model",
                "scaffold": "swelike",
                "model": "example",
                "reasoning_effort": None,
            },
            "run": {"index": 0, "attempt": "canonical", "retry_number": None},
            "prompts": [],
            "events": [
                {
                    "seq": 0,
                    "type": "tool_call",
                    "tool_call_id": "call-1",
                    "tool": "Read",
                    "input": {"path": "/repo/src/example.py"},
                },
                {
                    "seq": 1,
                    "type": "tool_result",
                    "tool_call_id": "call-1",
                    "status": "success",
                    "content": "example",
                },
            ],
            "usage": {},
            "evaluation": {
                "resolved": False,
                "status": "failed",
                "agent_exit_status": None,
                "evaluation_matches_selected": True,
            },
            "normalization": {
                "tool_result_limit_bytes": 65536,
                "truncated_tool_results": 0,
                "cache_tool_results_omitted": 0,
                "infrastructure_redactions": {},
            },
        }

        events, metrics, profile = parse_public_record(record, config)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["primary_operation"], "READ_FILE")
        self.assertEqual(events[0]["affected_paths"], ["src/example.py"])
        self.assertEqual(metrics["metrics"]["operational_action_count"], 1)
        self.assertEqual(profile["action_share__READ_FILE"], 1.0)
        projected = event_projection(record, events[0], "fable-5", 0)
        self.assertEqual(projected["contained_operations"], ["READ_FILE"])
        self.assertNotIn("command", projected)

    def test_public_program_has_no_machine_specific_absolute_path(self) -> None:
        text = (HERE / "multilingual_pipeline.py").read_text(encoding="utf-8")
        self.assertNotIn("/" + "home" + "/", text)
        self.assertNotIn("/" + "workspace" + "/", text)
        self.assertNotIn("parser_id", text)
        self.assertNotIn("codebook_version", text)


if __name__ == "__main__":
    unittest.main()
