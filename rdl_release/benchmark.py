"""Explicit model-forward timing; excludes preprocessing, NMS and data transfer."""
from __future__ import annotations

import platform
import statistics
import time
from pathlib import Path
from typing import Any


def benchmark_model(weights: Path, device_name: str, size: int, batch: int,
                    warmup: int, iterations: int) -> dict[str, Any]:
    """Benchmark a local trusted checkpoint with synchronized, gradient-free passes."""
    import torch
    from ultralytics import YOLO

    if size < 256 or size % 32:
        raise ValueError("Use an image size >=256 divisible by 32; the legacy FFT uses reflective padding")
    if batch < 1 or warmup < 0 or iterations < 1:
        raise ValueError("Batch and iterations must be positive; warmup must be nonnegative")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    model = YOLO(str(weights), task="detect").model.float().to(device).eval()
    tensor = torch.randn(batch, 3, size, size, device=device)

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    times: list[float] = []
    with torch.inference_mode():
        for _ in range(warmup):
            model(tensor)
        synchronize()
        for _ in range(iterations):
            synchronize()
            start = time.perf_counter()
            model(tensor)
            synchronize()
            times.append(time.perf_counter() - start)
    latency = statistics.mean(times)
    return {"protocol": "model_forward_only", "weights": weights.name, "precision": "FP32",
            "fusion_applied_by_runner": False, "checkpoint_fusion_state": "as_loaded",
            "device": str(device), "device_name":
            torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "torch": torch.__version__, "python": platform.python_version(),
            "cpu_threads": torch.get_num_threads(), "image_size": size, "batch": batch,
            "warmup": warmup, "iterations": iterations,
            "batch_latency_mean_ms": 1000 * latency,
            "batch_latency_std_ms": 1000 * statistics.pstdev(times),
            "image_latency_mean_ms": 1000 * latency / batch, "images_per_second": batch / latency,
            "includes_preprocessing": False, "includes_nms": False, "includes_transfer": False,
            "historical_paper_fps_reproduced": False}
