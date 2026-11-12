# Implementation details preserved for the release candidate

## Architecture

The full model applies RVB blocks in backbone and fusion stages, EMA in the selected deeper backbone blocks, the combined TSSA/DyT/Mona/EDFFN module after SPPF at backbone layer 10, and a shared LSCD/LQE head at strides 8/16/32. Configurations use explicit nano scale `n`. The A/B/C labels are architectural interventions for backbone/fusion RVB-EMA, SPPF-following frequency/statistics block, and shared calibrated head respectively. A therefore includes the supplied RVB fusion-stage replacements, rather than only the backbone layers.

## LQE channel grouping

The selected source preserves `pred_corners.reshape(B, reg_max, 4, H, W)` followed by softmax along `reg_max`. Conventional DFL decoding instead interprets raw channels as four contiguous blocks of `reg_max` logits. Thus the legacy estimator summarizes four interleaved channel groups rather than the four decoder boundary distributions. Top-k values and their means are concatenated into 4(k+1)=20 statistics; the two 1x1 layers use hidden size64 and produce a class-shared residual logit. The final layer starts at zero, so initial calibration is identity.

This distinction is intentional preservation, not endorsement of the per-boundary interpretation in the old submission. Reordering channels to conventional boundary-major distributions would change the model and require retraining/evaluation. The tests lock the actual selected operation to prevent silent scientific changes.

## Frequency and attention blocks

EDFFN uses gated convolutions and a learnable real-valued tensor applied to complex rFFT coefficients within local 8x8 patches. Reflection padding handles dimensions not divisible by8. It converts FFT inputs to FP32 and casts back to the original dtype. It is not a global full-feature complex-valued filter. TSSA applies L2-normalized squared token statistics, learned head temperatures and its original output projection. Mona includes multi-kernel depthwise convolutions and learned normalized/identity mixing. These details are represented by the source, not simplified substitute operators.

## Engine integration

The official Ultralytics 8.3.9 core is integrated with the selected operators. The parser registers their repeat/channel scaling, the shared head's scaled hidden channels, and detection stride initialization. Stride probing uses 2x640 images with backbone in eval mode and head producing training outputs, matching the selected research source's probing path. The release returns submodules to train mode after probing; normal train/eval commands explicitly set their appropriate mode. The core inherited C3k/C3k2/C2f/C3/Bottleneck/C2PSA/PSABlock/Attention/DFL ASTs match the supplied local source.

No broad research-module imports, unrelated custom CUDA extensions or inference-layout changes are needed. The original class/module paths remain available for selected-operator serialization. Compatibility with missing historical checkpoints is unverified, because checkpoint objects can include additional modified engine classes.
