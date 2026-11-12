"""Regression tests for command validation and empty upstream metric containers."""
import json
import unittest
from pathlib import Path

import numpy as np

from rdl_release.cli import evaluation_report, parser
from ultralytics.utils.metrics import DetMetrics


class CommandTests(unittest.TestCase):
    def test_empty_detection_metrics_serialize_without_nan(self):
        metrics = DetMetrics(names={0: "synthetic"})
        report = evaluation_report(metrics, Path("synthetic.pt"), "test")
        self.assertEqual(report["ap75"], 0.0)
        self.assertEqual(report["class_indices"], [])
        self.assertFalse(report["class_results_available"])
        json.dumps(report, allow_nan=False)

    def test_populated_detection_metrics_preserve_class_indices(self):
        metrics = DetMetrics(names={0: "first", 1: "second"})
        metrics.box.p = np.array([.4])
        metrics.box.r = np.array([.5])
        metrics.box.all_ap = np.full((1, 10), .3)
        metrics.box.ap_class_index = np.array([1])
        report = evaluation_report(metrics, Path("synthetic.pt"), "test")
        self.assertEqual(report["class_indices"], [1])
        self.assertAlmostEqual(report["ap75"], .3)
        self.assertTrue(report["class_results_available"])

    def test_train_defaults_match_documented_new_recipe(self):
        model = Path(__file__).resolve().parents[1] / "configs/models/rdl-yolo.yaml"
        args = parser().parse_args(["train", "--model", str(model), "--data", str(model)])
        self.assertEqual(args.epochs, 200)
        self.assertEqual(args.batch, 16)
        self.assertFalse(args.amp)
        self.assertIsNone(args.pretrained)


if __name__ == "__main__":
    unittest.main()
