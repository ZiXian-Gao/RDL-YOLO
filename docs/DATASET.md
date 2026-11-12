# NiP-DET dataset

The authors share NiP-DET through Quark Netdisk. On 12 November 2026, the share listed `NiP-DET.rar` (56.5 GB) and displayed a sign-in prompt in a signed-out browser. Full transfer, archive contents, and dataset licensing terms remain unverified. [NiP-DET on Quark Netdisk](https://pan.quark.cn/s/80f06f9a9557)

## Prepare the dataset for training

Use the original numeric class mapping and train/validation/test partitions from the dataset metadata. If the archive omits either, ask the dataset authors for the corresponding metadata before attempting a historical comparison. Use the paper's ten class names as category descriptions. Their order in the paper does not establish label IDs.

For Ultralytics, adapt the dataset YAML to your local paths and preserve the class IDs. The following is an example directory layout.

```text
NiP-DET/
  images/train/...
  images/val/...
  images/test/...
  labels/train/...
  labels/val/...
  labels/test/...
  data.yaml
```

Each nonempty annotation line is `class_id center_x center_y width height`, normalized to image dimensions. Empty labels represent negative images. Missing label files are also treated as negative by YOLO and are counted separately by the validator so that accidental omissions are visible.

The validator accepts split directories, image paths or text lists within the dataset root. Paths in a text list are relative to that list's parent or absolute. Use absolute paths or the conventional `./images/...` form in manifests consumed by both the validator and Ultralytics. Use an absolute dataset-root `path` to avoid upstream dataset-directory configuration ambiguity.

## Leakage and provenance

Run `python -m rdl_release validate-data --data /path/to/data.yaml`. The JSON includes relative paths, image/label hashes, per-split counts and per-class instances. Cross-split identical image content is an error. This check cannot identify every near duplicate, adjacent line-scan region, upper/lower face of one plate or same-batch dependency. Use a plate/batch acquisition-group audit before claiming independent generalization.

The manuscript reports 2,173 training, 271 validation, and 271 test images for NiP-DET. Those counts have not been verified against the shared archive. An independently created split is a new evaluation protocol. NEU-DET is a separate third-party dataset with its own distribution terms. The code license covers the software in this repository. Follow the applicable terms for each dataset.
