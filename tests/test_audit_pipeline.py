#!/usr/bin/env python3
"""Tests for model-panel audit summaries using synthetic labels."""
from __future__ import annotations

import json
import unittest

from audit_pipeline import FIELDS, summarize


class AuditPipelineTest(unittest.TestCase):
    def test_synthetic_consensus_and_scores(self):
        consensus, scores = [], []
        for event_id, label, prediction, weight in [("a", "A", "A", "1"), ("b", "B", "A", "2")]:
            event = {"audit_event_id": event_id, "sampling_weight": weight, "six_response_eligible": "true"}
            score = {"audit_event_id": event_id, "sampling_weight": weight}
            for field in FIELDS:
                event[f"vote_counts_{field}_json"] = json.dumps({label: 6})
                event[f"consensus_{field}"] = label
                score[f"{field}_reference"] = label
                score[f"{field}_prediction"] = prediction
                score[f"{field}_supported"] = "true"
                score[f"{field}_correct"] = str(label == prediction).lower()
            consensus.append(event)
            scores.append(score)
        result = summarize(consensus, scores)
        self.assertEqual(result["six_response_complete_events"], 2)
        primary = result["fields"]["primary_operation"]
        self.assertEqual(primary["supported"], 2)
        self.assertEqual(primary["fleiss_kappa"], 1.0)
        self.assertAlmostEqual(primary["sampling_weighted_accuracy"], 1 / 3)

    def test_duplicate_events_fail(self):
        event = {"audit_event_id": "duplicate"}
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize([event, event], [])


if __name__ == "__main__":
    unittest.main()
