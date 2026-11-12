"""Portable local-data entry points for training, evaluation and release checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _file(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Local file does not exist: {value}")
    return path


def _checkpoint(value: str) -> Path:
    path = _file(value)
    if path.suffix.lower() != ".pt":
        raise argparse.ArgumentTypeError("Use a local trusted .pt checkpoint")
    return path


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def evaluation_report(metrics: Any, weights: Path, split: str) -> dict[str, Any]:
    """Serialize upstream metrics, including its empty-list representation for no matches."""
    return {"weights": weights.name, "split": split,
            "metrics": {key: float(value) for key, value in metrics.results_dict.items()},
            "ap75": float(metrics.box.map75),
            "class_indices": [int(value) for value in metrics.box.ap_class_index],
            "class_names": metrics.names,
            "class_ap50": [float(value) for value in metrics.box.ap50],
            "class_ap50_95": [float(value) for value in metrics.box.ap],
            "class_results_available": bool(len(metrics.box.ap_class_index)),
            "timing_ms_per_image": metrics.speed,
            "historical_paper_results_reproduced": False}


def parser() -> argparse.ArgumentParser:
    """Construct the public command parser without loading the model runtime."""
    result = argparse.ArgumentParser(description="RDL-YOLO local research release tools")
    commands = result.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-data", help="Check labels, split overlap and image hashes")
    validate.add_argument("--data", type=_file, required=True)
    validate.add_argument("--output", type=Path, default=Path("dataset_validation.json"))
    for name in ("train", "evaluate", "predict", "benchmark"):
        command = commands.add_parser(name)
        command.add_argument("--device", default="cpu", help="cpu, 0, or cuda:0; benchmark uses torch device names")
        command.add_argument("--imgsz", type=int, default=640)
        if name == "train":
            command.add_argument("--model", type=_file, required=True)
            command.add_argument("--pretrained", type=_checkpoint)
            command.add_argument("--epochs", type=int, default=200)
            command.add_argument("--batch", type=int, default=16)
            command.add_argument("--seed", type=int, default=0)
            command.add_argument("--workers", type=int, default=0)
            command.add_argument("--amp", action="store_true", help="Enable AMP explicitly; requires validation on the target GPU")
        else:
            command.add_argument("--weights", type=_checkpoint, required=True,
                                 help="Only load checkpoints whose source you trust")
        if name in ("train", "evaluate"):
            command.add_argument("--data", type=_file, required=True)
        if name == "evaluate":
            command.add_argument("--split", choices=("val", "test"), default="test")
            command.add_argument("--batch", type=int, default=1)
            command.add_argument("--output", type=Path, default=Path("evaluation.json"))
        if name == "predict":
            command.add_argument("--source", type=Path, required=True)
            command.add_argument("--conf", type=float, default=0.25)
        if name in ("train", "evaluate", "predict"):
            command.add_argument("--project", type=Path, default=Path("runs"))
            command.add_argument("--name", default=name)
        if name == "benchmark":
            command.add_argument("--batch", type=int, default=1)
            command.add_argument("--warmup", type=int, default=20)
            command.add_argument("--iterations", type=int, default=100)
            command.add_argument("--output", type=Path, default=Path("benchmark.json"))
    return result


def _run(args: argparse.Namespace) -> int:
    if args.command == "validate-data":
        from .dataset import validate_dataset
        report = validate_dataset(args.data)
        _write_json(args.output, report)
        print(json.dumps({key: value for key, value in report.items() if key != "manifest"}, indent=2))
        return 0 if report["valid"] else 1
    if args.command == "benchmark":
        from .benchmark import benchmark_model
        device = "cuda:" + args.device if args.device.isdigit() else args.device
        report = benchmark_model(args.weights, device, args.imgsz, args.batch, args.warmup, args.iterations)
        _write_json(args.output, report)
        print(json.dumps(report, indent=2))
        return 0
    if args.imgsz < 256 or args.imgsz % 32:
        raise ValueError("Use imgsz >=256 divisible by 32 for the legacy FFT component")
    from ultralytics import YOLO
    common = {"imgsz": args.imgsz, "device": args.device,
              "project": str(args.project), "name": args.name}
    if args.command == "train":
        if args.model.suffix.lower() not in (".yaml", ".yml"):
            raise ValueError("Training --model must be a YAML architecture file")
        if args.epochs < 1 or args.batch < 1 or args.workers < 0:
            raise ValueError("Epochs and batch must be positive; workers nonnegative")
        model = YOLO(str(args.model), task="detect")
        if args.pretrained:
            model.load(str(args.pretrained))
        model.train(data=str(args.data), epochs=args.epochs, batch=args.batch,
                    seed=args.seed, workers=args.workers, optimizer="SGD", lr0=.01,
                    momentum=.937, weight_decay=.0005, pretrained=False, amp=args.amp,
                    deterministic=True, **common)
    elif args.command == "evaluate":
        if args.batch < 1:
            raise ValueError("Batch must be positive")
        model = YOLO(str(args.weights), task="detect")
        metrics = model.val(data=str(args.data), split=args.split, batch=args.batch, **common)
        _write_json(args.output, evaluation_report(metrics, args.weights, args.split))
    else:
        source = args.source.expanduser().resolve(strict=True)
        if not 0 <= args.conf <= 1:
            raise ValueError("Confidence threshold must lie in [0,1]")
        YOLO(str(args.weights), task="detect").predict(source=str(source), conf=args.conf, save=True, **common)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run a local command and emit concise validation errors."""
    arguments = parser()
    args = arguments.parse_args(argv)
    try:
        return _run(args)
    except (ValueError, FileNotFoundError, OSError) as error:
        arguments.exit(2, f"error: {error}\n")
