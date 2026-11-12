# Selected legacy RDL-YOLO operators. See provenance and THIRD_PARTY_NOTICES.md.
import torch
from torch import nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import C3k, C3k2
from ultralytics.nn.backbone.repvit import Conv2d_BN, RepVGGDW, SqueezeExcite
from .attention import EMA


class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class RepViTBlock(nn.Module):
    def __init__(self, inp, oup, use_se=True):
        super(RepViTBlock, self).__init__()

        self.identity = inp == oup
        hidden_dim = 2 * inp

        self.token_mixer = nn.Sequential(
            RepVGGDW(inp),
            SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
        )
        self.channel_mixer = Residual(nn.Sequential(
                # pw
                Conv2d_BN(inp, hidden_dim, 1, 1, 0),
                nn.GELU(),
                # pw-linear
                Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
            ))

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))


class RepViTBlock_EMA(RepViTBlock):
    def __init__(self, inp, oup, use_se=True):
        super().__init__(inp, oup, use_se)
        
        self.token_mixer = nn.Sequential(
            RepVGGDW(inp),
            EMA(inp) if use_se else nn.Identity(),
        )


class C3k_RVB(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepViTBlock(c_, c_, False) for _ in range(n)))


class C3k2_RVB(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_RVB(self.c, self.c, 2, shortcut, g) if c3k else RepViTBlock(self.c, self.c, False) for _ in range(n))


class C3k_RVB_EMA(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepViTBlock_EMA(c_, c_) for _ in range(n)))


class C3k2_RVB_EMA(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_RVB_EMA(self.c, self.c, 2, shortcut, g) if c3k else RepViTBlock_EMA(self.c, self.c) for _ in range(n))

