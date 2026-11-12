# Third party notices and permission record

The complete engine is based on the official Ultralytics 8.3.9 PyPI distribution. Its AGPL-3.0 license is retained in `LICENSE`; original headers and credits are preserved. Selected local research classes were extracted without changing their class bodies. The numerical operation graph has not been corrected or substituted for this release preparation.

| Component | Source or attribution | Evidence and retained material |
|---|---|---|
| Ultralytics engine | https://github.com/ultralytics/ultralytics | Official 8.3.9 wheel hash in `provenance/upstream_wheel.json`; AGPL-3.0 in root LICENSE |
| RepViT helpers | https://github.com/THU-MIG/RepViT | Apache-2.0; text in `licenses/RepViT-Apache-2.0.txt` |
| TSSA/ToST | https://github.com/RobinWu218/ToST | MIT; text in `licenses/ToST-MIT.txt` |
| EDFFN/EVSSM | https://github.com/kkkls/EVSSM | MIT; text in `licenses/EVSSM-MIT.txt` |
| Mona | https://github.com/Leiyi-Hu/mona | Mona-containing source file carries a Microsoft MIT header; surrounding OpenMMLab subtree carries Apache-2.0. Both notices retained below and in `licenses/`. Local adaptation uses NCHW convolutions. |
| EMA | https://github.com/YOLOonMe/EMA-attention-module | No explicit license identified in the checked public repository. The author confirmed permission to redistribute the local EMA adaptation on 4 November 2026; this confirmation is not an independently verified upstream license. |
| DynamicTanh | https://github.com/jiachenzhu/DyT | MIT license in `licenses/DyT-license.txt`. Credit: Transformers without Normalization, Zhu et al., CVPR 2025. The selected local implementation is identified in source provenance. |
| LQE principle | Generalized Focal Loss V2, Li et al., CVPR2021 | Distribution-statistics quality estimation is prior work. The local head adaptation and its legacy channel grouping are documented in `docs/IMPLEMENTATION.md`. |
| timm | https://github.com/huggingface/pytorch-image-models | Installed dependency, including SqueezeExcite; Apache-2.0. Preserve its own license when redistributing that dependency. |

## Mona source header retained

Source: `Swin-Transformer-Classification/mmcls/models/backbones/swin_transformer_mona.py` in the authors' repository:

```text
Swin Transformer
Copyright (c) 2021 Microsoft
Licensed under The MIT License [see LICENSE for details]
Written by Ze Liu, Yutong Lin, Yixuan Wei
```

The local Mona and EMA adaptations were supplied in the author's research source. The author confirmed redistribution permission for both adaptations. This record does not invent copyright ownership, a new upstream license, or a dataset license.

## Changes in this preparation

The release integration was prepared on 4 November 2026.

- Extracted selected classes into their existing legacy module paths; omitted unrelated research modules and optional CUDA extensions.
- Integrated the selected classes into the official 8.3.9 model parser and detection initialization/device handling.
- Added portable local-file commands, dataset checks, ablation configurations and synthetic tests.
- Preserved the original LQE grouping and selected operator bodies. Reconstructed combined ablation YAMLs are marked as such.

Release preparation does not remove the original rights or notices of third-party authors. See provenance/license_sources.json for retrieval URLs. The dataset and trained checkpoints have separate ownership/distribution terms and are not included here.
