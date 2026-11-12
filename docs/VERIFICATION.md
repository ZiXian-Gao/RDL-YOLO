# Code verification

On 12 November 2026, the release checkout passed all 11 tests on the same Windows CPU environment. The source comparison matched all 23 selected operator definitions and all nine inherited class definitions. The source distribution and wheel built successfully. Installing the wheel in an isolated environment with the tested CPU dependencies verified the `rdl-yolo` entry point and the packaged engine assets.

## Initial verification

Date: 4 November 2026, Asia/Shanghai.

## Outcome

The research code passed the checks below on Windows, Python 3.13.2, torch 2.11.0+cpu, torchvision 0.26.0+cpu, timm 1.0.7, einops 0.8.2 and numpy 2.2.3. The training and inference code uses the official Ultralytics 8.3.9 engine and preserves the selected research classes at their original module paths.

| Check | Result |
|---|---|
| Source fidelity | All23 extracted class ASTs and source-file hashes match the supplied research source; nine inherited core class ASTs also match. See source_fidelity.json. |
| Unit/regression suite | Eleven tests pass, including invalid labels, duplicate-image leakage, path escape, missing-label negatives and empty/nonempty evaluation metrics. |
| Architecture family | All eight reconstructed nano A/B/C configurations build and execute a real640x640 forward at ten classes. Detection strides are8/16/32; output is1x14x8400 with finite values. |
| Training loss | The full ten-class model computes finite detection loss and finite gradients on a synthetic640x640 batch. |
| Serialization | A locally generated trusted full model serializes and reloads with exactly equal evaluation outputs. |
| Legacy LQE | Sentinel logits verify the unchanged interleaved grouping; zero-initialized calibration is identity. |
| Data pipeline and trainer | One complete synthetic training epoch succeeds at256pixels with two generated training images and two validation images. Checkpoints are produced and reloaded. |
| Public wrappers | Checkpoint evaluation at640pixels, local-image prediction, and synchronized model-forward benchmark run successfully on synthetic inputs. |
| Build/import | A wheel builds successfully. Its package and CLI import from an installation into a temporary environment with system site packages. This validates packaging against available dependencies, not an independently provisioned full dependency environment. |
| Syntax | New helper/test files and selected modules compile. |

The observed direct-dependency versions are in requirements-tested.txt. Combined ablation YAMLs are structurally reconstructed from the supplied single-component designs and are not represented as recovered historical run artifacts.

## Repairs found during verification

The first dataset validation test exposed an unclosed hashing stream; it was corrected to use a context manager. End-to-end evaluation exposed upstream metrics represented as empty lists when no valid detections exist; the wrapper now uses the upstream AP75 property and serializes empty class arrays correctly. Both repairs affect release helpers, not the selected detector computation. Explicit YAML scale is retained when no model-size suffix is present. The selected legacy class ASTs remain unchanged after these repairs.

## Scope limits

- No original NiP-DET/NEU release data, exact class map, trained research weights, original split manifests or complete historical logs were available for this check.
- No manuscript accuracy/FPS result was reproduced or replaced. Generated images and random or one-epoch synthetic weights establish executable paths only.
- CPU verification does not establish A30/GPU, AMP, multi-GPU, ONNX/TensorRT export, industrial latency or cross-platform dependency compatibility.
- Selected class paths are retained, but missing historical checkpoint files may contain additional modified engine classes. Their compatibility remains untested.
- Full-source AST equality establishes code extraction fidelity; it does not establish historical experiment identity, random-seed initialization equivalence across environments, dataset independence or correct scientific interpretation.
- The original LQE reshape is preserved and differs from conventional boundary-major DFL grouping. Correcting it would require a separately identified model and new experiments.
- Mona/EMA redistribution permission is recorded as author-confirmed. Verified upstream notices remain included; no additional public license is invented.

