#!/usr/bin/env python3
"""Tests for public Python panel extraction and task-equal aggregation."""
from __future__ import annotations

import json
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from python_pipeline import METRICS, aggregate, build_tables, parquet_records, parse_record, public_files, safe_root, unwrap_codex_shell
from python_command_methods import classify, explicit_pytest_quiet


class PythonPipelineTest(unittest.TestCase):
    def record(self):
        events = [
            {"seq": 0, "type": "tool_call", "tool_call_id": "a", "tool": "Read", "input": {"path": "/repo/example.py"}},
            {"seq": 1, "type": "tool_result", "tool_call_id": "a", "status": "success", "content": "private payload"},
            {"seq": 2, "type": "tool_call", "tool_call_id": "b", "tool": "Bash", "input": {"command": "pytest -q"}},
            {"seq": 3, "type": "tool_result", "tool_call_id": "b", "status": "success", "content": "1 passed"},
        ]
        return {"trajectory_id": "trajectory", "task_id": "task", "repository": "owner/repo",
                "agent_config_id": "config", "canonical_model": "model", "scaffold": "test",
                "reasoning_effort": None, "run_index": 0, "resolved": None,
                "trajectory": json.dumps({"schema_version": "1.1.0", "events": events})}

    def test_parse_and_redaction(self):
        result = parse_record(self.record())
        self.assertEqual(result["event_count"], 2)
        self.assertEqual(result["metrics"]["read_share"], 0.5)
        self.assertEqual(result["metrics"]["native_read_share"], 1.0)
        self.assertEqual(result["metrics"]["quiet_pytest_share"], 1.0)
        self.assertIsNone(result["metrics"]["python_edit_share"])
        self.assertIsNone(result["resolved"])
        self.assertNotIn("private payload", json.dumps(result))
        self.assertNotIn("pytest -q", json.dumps(result))

    def test_unsupported_file_changes_are_not_aggregated(self):
        record = self.record()
        trajectory = json.loads(record["trajectory"])
        trajectory["events"].append({"seq": 4, "type": "file_change", "changes": [{"path": "example.py"}]})
        record["trajectory"] = json.dumps(trajectory)
        result = parse_record(record)
        self.assertEqual(result["unparsed_file_changes"], 1)
        profiles = {r["metric"]: r for r in aggregate([result])}
        self.assertIsNone(profiles["read_share"]["task_equal_mean"])
        self.assertEqual(profiles["read_share"]["rollouts_excluded_unparsed_file_change"], 1)

    def test_command_method_classification(self):
        self.assertEqual(classify("Bash", "sed -n '1,10p' app.py", "READ_SOURCE")["read_method"], "sed_print")
        self.assertEqual(classify("Bash", "python3 -c 'print(1)'", "EDIT_SOURCE")["edit_method"], "python_inline_mutation")
        self.assertEqual(classify("Bash", "echo text > app.py", "EDIT_SOURCE")["edit_method"], "shell_redirection")
        self.assertTrue(explicit_pytest_quiet("pytest -q tests", "pytest_direct"))
        self.assertFalse(explicit_pytest_quiet("pytest -- -q", "pytest_direct"))

    def test_shell_wrapper_decode_fails_closed(self):
        self.assertEqual(unwrap_codex_shell("/bin/bash -lc 'python3 -c hello'"), ("python3 -c hello", True))
        self.assertEqual(unwrap_codex_shell("bash -c 'hello' extra"), ("bash -c 'hello' extra", False))
        self.assertFalse(unwrap_codex_shell("bash -c 'unterminated")[1])

    def test_integer_outcomes_are_strictly_normalized(self):
        for value, expected in [(0, False), (1, True), (None, None)]:
            record = self.record()
            record["resolved"] = value
            self.assertIs(parse_record(record)["resolved"], expected)
        record["resolved"] = 2
        with self.assertRaises(ValueError):
            parse_record(record)

    def test_hf_parquet_aliases(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow is an optional Python-panel dependency")
        record = self.record()
        record["instance_id"] = record.pop("task_id")
        record["repo"] = record.pop("repository")
        record["resolved"] = 1
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.parquet"
            pq.write_table(pa.Table.from_pylist([record]), path)
            result = parse_record(next(iter(parquet_records([path]))))
        self.assertEqual(result["task_id"], "task")
        self.assertEqual(result["repository"], "owner/repo")
        self.assertIs(result["resolved"], True)

    def test_hf_download_resolves_revision_and_selects_python_parquet(self):
        calls = {}
        class FakeApi:
            def dataset_info(self, dataset, revision):
                calls["info"] = (dataset, revision)
                return types.SimpleNamespace(sha="immutable-sha")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            file = root / "data" / "python_spring2026" / "train.parquet"
            file.parent.mkdir(parents=True)
            file.touch()
            def snapshot_download(**kwargs):
                calls["snapshot"] = kwargs
                return str(root)
            module = types.SimpleNamespace(HfApi=FakeApi, snapshot_download=snapshot_download)
            with patch.dict(sys.modules, {"huggingface_hub": module}):
                files, sha = public_files("requested", root / "cache")
            self.assertEqual(files, [file])
            self.assertEqual(sha, "immutable-sha")
            self.assertEqual(calls["info"][1], "requested")
            self.assertEqual(calls["snapshot"]["revision"], "immutable-sha")
            self.assertEqual(calls["snapshot"]["allow_patterns"], "data/python_spring2026/*.parquet")

    def test_missing_root_fails(self):
        with self.assertRaises(ValueError):
            safe_root("")
        with self.assertRaises(ValueError):
            safe_root("repo")

    def test_task_equal_aggregation_preserves_missingness(self):
        rows = []
        for trajectory_id, task_id, value in [("a", "one", 0), ("b", "one", 1), ("c", "two", 1)]:
            row = parse_record(self.record())
            row["trajectory_id"] = trajectory_id
            row["task_id"] = task_id
            row["metrics"] = {metric: None for metric in METRICS}
            row["metrics"]["read_share"] = value
            rows.append(row)
        profiles = {r["metric"]: r for r in aggregate(rows)}
        self.assertEqual(profiles["read_share"]["task_equal_mean"], 0.75)
        self.assertEqual(profiles["read_share"]["tasks_defined"], 2)
        self.assertEqual(profiles["read_share"]["rollouts_defined"], 3)
        self.assertIsNone(profiles["edit_share"]["task_equal_mean"])
        self.assertEqual(profiles["edit_share"]["tasks_defined"], 0)

    def test_table_projection_preserves_missing_and_partial_support(self):
        config = {
            "generation_effort": [{"label": "Present", "agent_config_id": "config"}, {"label": "Missing", "agent_config_id": "missing"}],
            "scaffolds": [{"label": "Present", "agent_config_id": "config"}],
            "generation_effort_metrics": ["read_share"], "scaffold_metrics": ["edit_share"],
        }
        record = parse_record(self.record())
        unsupported = dict(record, trajectory_id="other", unparsed_file_changes=1)
        tables = build_tables(aggregate([record, unsupported]), config)
        self.assertEqual(tables["generation_effort"][0]["status"], "partial_file_changes_excluded")
        self.assertEqual(tables["generation_effort"][0]["rollouts_excluded_unparsed_file_change"], 1)
        self.assertEqual(tables["generation_effort"][1]["status"], "missing_configuration")
        self.assertIsNone(tables["generation_effort"][1]["task_equal_mean"])
        self.assertEqual(tables["scaffold"][0]["metric"], "edit_share")

    def test_inconsistent_stack_metadata_fails(self):
        first = parse_record(self.record())
        second = dict(first, scaffold="other")
        with self.assertRaisesRegex(ValueError, "inconsistent metadata"):
            aggregate([first, second])

    def test_python_analysis_config_has_complete_table_selections(self):
        config_path = Path(__file__).resolve().parents[1] / "config" / "python_analysis.json"
        config = json.loads(config_path.read_text())
        tables = build_tables([], config)
        self.assertEqual(len(tables["generation_effort"]), 12 * 7)
        self.assertEqual(len(tables["scaffold"]), 4 * 4)
        self.assertTrue(all(row["status"] == "missing_configuration" for rows in tables.values() for row in rows))

    def test_python_analysis_files_have_simple_names(self):
        root = Path(__file__).resolve().parents[1]
        files = [*root.glob("python*.py"), *(root / "python_historical").rglob("*.py")]
        self.assertTrue(files)
        for path in files:
            self.assertNotRegex(path.name, re.compile(r"(?:^|_)v\d+", re.I))

    def test_no_machine_paths_in_release_code(self):
        source = (Path(__file__).resolve().parents[1] / "python_pipeline.py").read_text()
        self.assertNotIn("/" + "home" + "/", source)
        self.assertNotIn("/" + "Users" + "/", source)


if __name__ == "__main__":
    unittest.main()
