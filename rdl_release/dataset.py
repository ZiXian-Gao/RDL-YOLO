"""Validate YOLO detection data without altering images, labels or splits."""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Path outside dataset root: {path}")
    return resolved


def _split_images(value: Any, root: Path) -> list[Path]:
    entries = value if isinstance(value, list) else [value]
    images: list[Path] = []
    for entry in entries:
        if not isinstance(entry, str) or not entry.strip():
            raise ValueError("Each split must name a directory, image or text manifest")
        path = _inside(root / entry, root)
        if path.is_dir():
            images.extend(sorted(_inside(item, root) for item in path.rglob("*")
                                 if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES))
        elif path.is_file() and path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                if line.strip():
                    images.append(_inside(path.parent / line.strip(), root))
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path)
        else:
            raise ValueError(f"Split path does not exist or is unsupported: {path}")
    return images


def _class_names(value: Any) -> list[str]:
    if isinstance(value, dict):
        try:
            ordered = {int(key): name for key, name in value.items()}
        except (ValueError, TypeError) as error:
            raise ValueError("Class names mapping must use integer IDs") from error
        if len(ordered) != len(value) or sorted(ordered) != list(range(len(ordered))):
            raise ValueError("Class IDs must be unique and consecutive from zero")
        names = [ordered[index] for index in range(len(ordered))]
    elif isinstance(value, list):
        names = value
    else:
        raise ValueError("Provide the actual ordered class names in the dataset YAML")
    if not names or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Class names must be nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("Class names must be unique")
    if any("replace_with" in name for name in names):
        raise ValueError("Replace example class names with the dataset's actual label order")
    return names


def _label_path(image: Path, root: Path) -> Path:
    parts = list(image.relative_to(root).parts)
    if "images" not in parts:
        raise ValueError(f"Expected an images directory in image path: {image.relative_to(root)}")
    parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
    return _inside(root.joinpath(*parts).with_suffix(".txt"), root)


def _labels(path: Path, class_count: int) -> list[int]:
    classes: list[int] = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path.name}:{number}: expected class x_center y_center width height")
        values = [float(field) for field in fields]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path.name}:{number}: nonfinite label")
        category, center_x, center_y, width, height = values
        if not category.is_integer() or not 0 <= category < class_count:
            raise ValueError(f"{path.name}:{number}: class ID outside dataset names")
        if not (0 <= center_x <= 1 and 0 <= center_y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"{path.name}:{number}: coordinates must be normalized with positive sizes")
        if (center_x - width / 2 < -1e-5 or center_y - height / 2 < -1e-5
                or center_x + width / 2 > 1 + 1e-5 or center_y + height / 2 > 1 + 1e-5):
            raise ValueError(f"{path.name}:{number}: box extends outside image")
        classes.append(int(category))
    return classes


def validate_dataset(config: str | Path) -> dict[str, Any]:
    """Return counts, split/content-overlap errors and a relative-path SHA256 manifest.

    Missing labels follow YOLO's negative-image convention and are counted explicitly.
    Byte-identical image checks cannot detect all related captures or near duplicates.
    """
    config = Path(config).resolve(strict=True)
    data = yaml.safe_load(config.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("Dataset YAML must contain a mapping")
    names = _class_names(data.get("names"))
    if "nc" in data and data["nc"] != len(names):
        raise ValueError("nc does not match the number of ordered class names")
    root = (config.parent / data.get("path", ".")).resolve(strict=True)
    report: dict[str, Any] = {"valid": True, "class_names": names, "splits": {},
                              "errors": [], "warnings": [], "manifest": []}
    seen_paths: dict[Path, str] = {}
    seen_hashes: dict[str, tuple[str, str]] = {}
    for split in ("train", "val", "test"):
        if split not in data:
            report["warnings"].append(f"No {split} split declared")
            continue
        try:
            images = _split_images(data[split], root)
        except (ValueError, OSError) as error:
            report["errors"].append(f"{split}: {error}")
            continue
        counts: Counter[int] = Counter()
        negative_count = missing_count = 0
        if not images:
            report["errors"].append(f"{split}: no images")
        for image in images:
            relative = image.relative_to(root).as_posix()
            try:
                if image in seen_paths:
                    raise ValueError(f"Image path repeated in {seen_paths[image]} and {split}: {relative}")
                seen_paths[image] = split
                with Image.open(image) as opened:
                    opened.verify()
                with image.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest in seen_hashes and seen_hashes[digest][0] != split:
                    other_split, other_image = seen_hashes[digest]
                    raise ValueError(f"Byte-identical image in {other_split}/{split}: {other_image}, {relative}")
                seen_hashes[digest] = (split, relative)
                label = _label_path(image, root)
                classes = _labels(label, len(names)) if label.exists() else []
                missing_count += int(not label.exists())
                negative_count += int(not classes)
                counts.update(classes)
                label_hash = None
                if label.exists():
                    label_hash = hashlib.sha256(label.read_bytes()).hexdigest()
                report["manifest"].append({"split": split, "image": relative, "sha256": digest,
                                           "label": label.relative_to(root).as_posix(),
                                           "label_sha256": label_hash, "instances": len(classes)})
            except (ValueError, OSError) as error:
                report["errors"].append(f"{split}/{relative}: {error}")
        report["splits"][split] = {"images": len(images), "instances": sum(counts.values()),
                                    "instances_by_class": {str(i): counts[i] for i in range(len(names))},
                                    "negative_images": negative_count, "missing_labels": missing_count}
    if not report["splits"]:
        report["errors"].append("No usable train, val or test split")
    report["valid"] = not report["errors"]
    return report
