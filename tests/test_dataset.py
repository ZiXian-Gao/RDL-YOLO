"""Tests use generated images and labels, never the research dataset."""
import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image

from rdl_release.dataset import validate_dataset


class DatasetValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for index, split in enumerate(("train", "val", "test")):
            image_dir = self.root / "images" / split
            label_dir = self.root / "labels" / split
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            Image.new("RGB", (20, 16), (index * 70, 0, 0)).save(image_dir / "sample.png")
            (label_dir / "sample.txt").write_text("0 0.5 0.5 0.2 0.3\n", encoding="utf-8")
        self.config = self.root / "data.yaml"
        self.config.write_text(yaml.safe_dump({
            "path": str(self.root), "train": "images/train", "val": "images/val",
            "test": "images/test", "names": ["synthetic_class"],
        }), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_valid_dataset_and_manifest(self):
        report = validate_dataset(self.config)
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["splits"]["test"]["instances"], 1)
        self.assertEqual(report["class_names"], ["synthetic_class"])
        self.assertEqual(len(report["manifest"]), 3)

    def test_cross_split_duplicate_content_fails(self):
        (self.root / "images/val/sample.png").write_bytes(
            (self.root / "images/train/sample.png").read_bytes())
        report = validate_dataset(self.config)
        self.assertFalse(report["valid"])
        self.assertTrue(any("identical image" in error for error in report["errors"]))

    def test_out_of_range_class_and_box_fail(self):
        (self.root / "labels/test/sample.txt").write_text("1 0.9 0.5 0.8 0.3\n")
        report = validate_dataset(self.config)
        self.assertFalse(report["valid"])
        self.assertTrue(any("class" in error for error in report["errors"]))

    def test_missing_labels_are_counted_as_negatives(self):
        (self.root / "labels/test/sample.txt").unlink()
        report = validate_dataset(self.config)
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["splits"]["test"]["missing_labels"], 1)
        self.assertEqual(report["splits"]["test"]["negative_images"], 1)

    def test_symlink_or_manifest_escape_fails(self):
        listing = self.root / "train.txt"
        listing.write_text("../outside.png\n", encoding="utf-8")
        data = yaml.safe_load(self.config.read_text())
        data["train"] = "train.txt"
        self.config.write_text(yaml.safe_dump(data))
        report = validate_dataset(self.config)
        self.assertFalse(report["valid"])
        self.assertTrue(any("outside dataset root" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
