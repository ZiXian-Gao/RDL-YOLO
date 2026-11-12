# RDL-YOLO for nickel plate defect detection

RDL-YOLO detects surface defects on nickel plates. This repository provides the research implementation based on Ultralytics 8.3.9, model and ablation configurations, commands for training and evaluation, and synthetic verification tests.

The authors share NiP-DET separately through [Quark Netdisk](https://pan.quark.cn/s/80f06f9a9557). This release has been verified with synthetic inputs on CPU. The [reproducibility notes](docs/REPRODUCIBILITY.md) describe the evidence available for the manuscript's reported accuracy and throughput.

## Installation

Use a fresh environment. This checkout provides the `ultralytics` import package, so do not install a second Ultralytics distribution into the same environment. Python 3.13.2 with torch 2.11.0 CPU was tested locally; Python >=3.11 is required by the release helpers. The paper's earlier Python/PyTorch environment is a separate historical claim.

```bash
uv venv --python 3.13
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate

uv pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install -e .
python -m rdl_release --help
```

For GPU execution, install the matching torch/torchvision build from the official PyTorch installer before the editable install. GPU execution, mixed precision and export engines were not validated in the local CPU checks. [requirements-tested.txt](requirements-tested.txt) records tested direct-dependency versions rather than claiming a cross-platform lockfile.

## Dataset preparation

Download NiP-DET from the linked share and follow the [dataset documentation](docs/DATASET.md). Use the original numeric class mapping and split definitions. If the archive omits either, request the metadata from the dataset authors. `configs/dataset.example.yaml` shows the expected structure and uses a placeholder class. Set `path` to the absolute dataset root for both the validator and trainer.

```bash
python -m rdl_release validate-data --data /path/to/data.yaml --output dataset_validation.json
```

The validator checks image decoding, five-column YOLO boxes, class ranges, empty/missing labels, duplicate file paths, and byte-identical images across splits. It does not establish plate-level independence or detect every near duplicate.

## Train

Run commands from this repository root after installation:

```bash
python -m rdl_release train --model configs/models/rdl-yolo.yaml --data /path/to/data.yaml --device cpu --epochs 200 --batch 16 --seed 0
```

`--device 0` selects a GPU. Training defaults to random initialization, SGD, lr0=0.01, momentum=0.937 and weight_decay=0.0005. AMP is disabled unless `--amp` is supplied. To initialize from an existing trusted checkpoint, add `--pretrained /path/to/checkpoint.pt`. These are **documented new-run settings**, not a recovered complete historical recipe. Other augmentation and scheduler settings follow the included `ultralytics/cfg/default.yaml` and are saved by the trainer in each run's `args.yaml`.

The model YAML's generic `nc=80` is overridden by the dataset's actual class count during training. Explicitly use nano scale `n` for the paper's architecture. The complete 2^3 configuration family is in `configs/ablations/`; combined variants are reconstructed from the supplied single-component configurations and are not asserted to be the missing original run files.

## Evaluate and predict

```bash
python -m rdl_release evaluate --weights /path/to/best.pt --data /path/to/data.yaml --split test --device cpu --output evaluation.json
python -m rdl_release predict --weights /path/to/best.pt --source /path/to/images --device cpu
```

The evaluation command writes AP50, AP75, AP50:95, per-class AP and component timing. It does not silently replace the manuscript's historical table. Checkpoints are Python-serialized objects; load only checkpoints from a trusted source. No checkpoints are downloaded automatically by these wrappers.

## Benchmark

```bash
python -m rdl_release benchmark --weights /path/to/best.pt --device cpu --batch 1 --imgsz 640 --warmup 20 --iterations 100 --output benchmark.json
```

This measures synchronized FP32 **model-forward latency only** on a synthetic tensor, with inference mode enabled and explicit warm-up. It excludes acquisition, resizing, transfers, NMS and sorting. It is a new documented protocol; it does not recover the original manuscript's incompletely specified FPS protocol. A default 640-pixel input is supported. Very small input sizes can fail the original patchwise FFT's reflective padding; the wrapper requires a multiple of 32 at least 256 pixels.

## Verification

```bash
python -m unittest discover -s tests -v
```

Tests use generated images, random weights and synthetic labels. They test actual 640-pixel model forwards for all eight configurations, finite training loss/backward, serialization, and the unchanged legacy LQE grouping. No dataset benchmark accuracy is implied.

## Method and provenance

The frequency block follows SPPF at backbone layer 10. It is not repeated after every neck fusion. The three head inputs correspond to strides 8, 16 and 32. EDFFN uses 8x8 patchwise real FFTs with real-valued learned multipliers. TSSA uses L2 normalization. The historical LQE reshape groups channels as `[B,16,4,H,W]`; this differs from conventional per-boundary DFL grouping and is deliberately preserved. See [implementation notes](docs/IMPLEMENTATION.md).

The distribution is based on the official PyPI Ultralytics 8.3.9 wheel plus selected definitions from the author-provided source. [Provenance records](provenance/) identify source hashes and extracted definitions. This documents code preparation, not the unknown historical experiment snapshot.

## License and acknowledgments

The engine is distributed under AGPL-3.0-or-later. Preserve [LICENSE](LICENSE), [third-party notices](THIRD_PARTY_NOTICES.md), and [upstream license files](licenses/). The project reuses/adapts RepViT, EMA, ToST/TSSA, DynamicTanh, Mona, EVSSM/EDFFN and localization-quality estimation ideas. The author confirmed permission to redistribute the local Mona/EMA adaptations; this is recorded separately from independently verified upstream licenses.

## Citation

Use [CITATION.cff](CITATION.cff) to cite the research code.
