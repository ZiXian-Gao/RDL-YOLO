# Reproducibility status

## Verified code paths

The included engine and selected operators can be imported, architectures built, and synthetic forward/training/serialization paths checked on the stated local CPU runtime. Exact extraction provenance and code-body fidelity are recorded separately from experiment provenance. The release contains all eight structurally reconstructed A/B/C configurations.

## Historical benchmark evidence

The authors provide a [NiP-DET share link](https://pan.quark.cn/s/80f06f9a9557). The release checks used synthetic inputs. The archive's class-ID map and split manifests remain unverified, and this release does not provide trained research weights or complete original benchmark logs. The manuscript's accuracy and FPS figures come from the reported historical experiments on an NVIDIA A30 with Python 3.8 and PyTorch 1.13.1. See [verification](VERIFICATION.md) for the separate CPU check environment.

The runner exposes a concrete documented recipe based on available manuscript hyperparameters. Unreported historical augmentation, initialization, seed, early-stopping, software changes and timing choices cannot be reconstructed from the paper alone. Save run arguments, environment versions, data manifests, code commit and checkpoint hashes with every new run. Distinguish new validation from submitted results.

## Next reproducible release

1. Confirm the numeric class mapping, dataset terms and original split manifests against the shared archive or the original dataset metadata.
2. Associate each manuscript row with its original configuration, checkpoint and validation output, if preserved.
3. Validate released splits and check plate/batch independence separately.
4. Rerun trained checkpoints on the frozen test sets, exporting AP50/AP75/AP50:95 and per-class metrics.
5. Repeat key training configurations with declared seeds if scientific claims depend on stability.
6. Report GPU target, precision, fusion, warm-up and inclusion/exclusion of preprocessing/NMS for all new timing.

The existing historical results are not evidence that any newly generated configuration reproduces the same run. No synthetic performance values should be copied into the manuscript.
