# Selected legacy RDL-YOLO operators. See provenance and THIRD_PARTY_NOTICES.md.
import torch
from torch import nn
import torch.nn.functional as F
import math
from torch.nn import init
from ultralytics.nn.modules import DFL
from ultralytics.nn.modules.conv import autopad
from ultralytics.utils.tal import dist2bbox, make_anchors


class Scale(nn.Module):
    """A learnable scale parameter.

    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.

    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class Conv_GN(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.gn = nn.GroupNorm(16, c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.gn(self.conv(x)))


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        # self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.layers = nn.ModuleList(nn.Conv2d(n, k, 1) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = nn.ReLU()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LQE(nn.Module):
    """
    位置质量估计器 (Location Quality Estimator, LQE)
    用于评估和调整边界框预测的质量分数，结合分布统计信息提升精度
    """
    def __init__(self, k, hidden_dim, num_layers, reg_max):
        """
        初始化 LQE 模块
        参数:
            k: 前 k 个最高概率值的数量，用于统计分析
            hidden_dim: MLP隐藏层维度
            num_layers: MLP层数
            reg_max: 回归的最大值（边界框分布的最大范围）
        """
        super(LQE, self).__init__()
        self.k = k
        self.reg_max = reg_max
        # 定义一个多层感知机（MLP），输入维度为 4*(k+1)，输出为 1
        self.reg_conf = MLP(4 * (k + 1), hidden_dim, 1, num_layers)
        # 初始化最后一层的偏置和权重为 0
        init.constant_(self.reg_conf.layers[-1].bias, 0)
        init.constant_(self.reg_conf.layers[-1].weight, 0)

    def forward(self, scores, pred_corners):
        """
        前向传播
        参数:
            scores: 初始分类得分 [B, num_classes, h, w]
            pred_corners: 预测的边界框角点分布 [B, 4*(reg_max), h, w]
        返回:
            调整后的质量分数
        """
        # 计算 softmax 概率
        B, C, H, W = pred_corners.size()
        prob = F.softmax(pred_corners.reshape(B, self.reg_max, 4, H, W), dim=1)
        # 提取前 k 个最高概率值及其索引
        prob_topk, _ = prob.topk(self.k, dim=1)
        # 将 top-k 概率及其均值拼接，作为统计特征
        stat = torch.cat([prob_topk, prob_topk.mean(dim=1, keepdim=True)], dim=1)
        # 通过 MLP 计算质量分数调整值
        quality_score = self.reg_conf(stat.reshape(B, -1, H, W))
        # 将初始得分与质量调整值相加
        return scores + quality_score


class Detect_LSCD_LQE(nn.Module):
    # Lightweight Shared Convolutional Detection Head
    """YOLOv8 Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, hidc=256, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        self.conv = nn.ModuleList(nn.Sequential(Conv_GN(x, hidc, 3)) for x in ch)
        self.share_conv = nn.Sequential(Conv_GN(hidc, hidc, 3, g=hidc), Conv_GN(hidc, hidc, 1))
        self.cv2 = nn.Conv2d(hidc, 4 * self.reg_max, 1)
        self.cv3 = nn.Conv2d(hidc, self.nc, 1)
        self.scale = nn.ModuleList(Scale(1.0) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        self.lqe = nn.ModuleList(LQE(4, 64, 2, self.reg_max) for x in ch)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = self.conv[i](x[i])
            x[i] = self.share_conv(x[i])
            pred_corners = self.scale[i](self.cv2(x[i]))
            pred_scores = self.lqe[i](self.cv3(x[i]), pred_corners)
            x[i] = torch.cat((pred_corners, pred_scores), 1)
        if self.training:  # Training path
            return x

        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(box)

        if self.export and self.format in ("tflite", "edgetpu"):
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            img_h = shape[2]
            img_w = shape[3]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * img_size)
            dbox = dist2bbox(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2], xywh=True, dim=1)

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        # for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
        m.cv2.bias.data[:] = 1.0  # box
        m.cv3.bias.data[: m.nc] = math.log(5 / m.nc / (640 / 16) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes):
        """Decode bounding boxes."""
        return dist2bbox(self.dfl(bboxes), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

