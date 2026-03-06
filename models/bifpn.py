# ------------------------------------------------------------------------
# BiFPN: Bidirectional Feature Pyramid Network
# 双向特征金字塔网络
# ------------------------------------------------------------------------
# 参考: EfficientDet 官方实现
# https://github.com/zylox117/EfficientDet-Pytorch
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Conv2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow conv2d with same padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, dilation=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=bias, groups=groups)
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]

        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.conv(x)
        return x


class MaxPool2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow MaxPool2d with same padding
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(*args, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]

        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.pool(x)
        return x


class SeparableConvBlock(nn.Module):
    """
    created by Zylo117
    深度可分离卷积块
    """

    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels,
                                                      kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x


class BiFPN(nn.Module):
    """
    4 层 BiFPN: 处理 P3, P4, P5, P6

    数据流 (对应官方7层架构，P6对应官方P7位置):
        Top-Down:                         Bottom-Up:
        P6_in (最高层，不生成td)           P3_td → P3_out
          ↓ (上采样)                         ↑ (下采样)
        P5_in + P6_in↑ → P5_td             P4_td + P4_in + P3_out↓ → P4_out
          ↓ (上采样)                         ↑ (下采样)
        P4_in + P5_td↑ → P4_td             P5_td + P5_in + P4_out↓ → P5_out
          ↓ (上采样)                         ↑ (下采样)
        P3_in + P4_td↑ → P3_td             P6_in + P5_out↓ → P6_out

    注意: P6 是最高层（对应官方的P7）
          - Top-Down: P6_in 不生成中间层，直接上采样给P5使用
          - Bottom-Up: P6_out 只用 2 个输入 (P6_in + P5_out↓)，与官方P7一致
    """

    def __init__(self, num_channels=256, epsilon=1e-4, onnx_export=False, attention=True):
        """
        Args:
            num_channels: 统一的特征通道数
            epsilon: 快速归一化求和的小常数
            onnx_export: 是否导出 ONNX
            attention: 是否使用快速注意力模式
        """
        super(BiFPN, self).__init__()
        self.epsilon = epsilon
        self.onnx_export = onnx_export
        self.attention = attention

        # ========== Top-Down 卷积层 (用于生成中间层 _td) ==========
        # 注意：最高层P6不生成中间层，直接用于上采样
        self.conv5up = SeparableConvBlock(num_channels, onnx_export=onnx_export)   # P5_td
        self.conv4up = SeparableConvBlock(num_channels, onnx_export=onnx_export)   # P4_td
        self.conv3up = SeparableConvBlock(num_channels, onnx_export=onnx_export)   # P3_td

        # ========== Bottom-Up 卷积层 (用于生成输出层 _out，与官方命名一致) ==========
        self.conv3down = SeparableConvBlock(num_channels, onnx_export=onnx_export)  # P3_out (直接输出，实际不使用)
        self.conv4down = SeparableConvBlock(num_channels, onnx_export=onnx_export)  # P4_out
        self.conv5down = SeparableConvBlock(num_channels, onnx_export=onnx_export)  # P5_out
        self.conv6down = SeparableConvBlock(num_channels, onnx_export=onnx_export)  # P6_out

        # ========== 下采样层 ==========
        self.p4_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p5_downsample = MaxPool2dStaticSamePadding(3, 2)

        # Swish 激活函数
        if not self.onnx_export:
            self.swish = MemoryEfficientSwish()
        else:
            self.swish = Swish()

        # ========== 可学习权重 (Top-Down) ==========
        # P6: 最高层，不生成中间层，直接用于上采样
        # P5 ← P5 + P6↑
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1_relu = nn.ReLU()
        # P4 ← P4 + P5↑
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        # P3 ← P3 + P4↑
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()

        # ========== 可学习权重 (Bottom-Up) ==========
        # P4 ← P4 + P4_td + P3_out↓
        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        # P5 ← P5 + P5_td + P4_out↓
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()
        # P6 ← P6 + P5_out↓ (最高层，只有2个输入，不使用P6_td)
        self.p6_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p6_w2_relu = nn.ReLU()

    def forward(self, inputs):
        """
        Args:
            inputs: list of tensors [p3_in, p4_in, p5_in, p6_in]

        Returns:
            outs: list of tensors [p3_out, p4_out, p5_out, p6_out]
        """
        if self.attention:
            outs = self._forward_fast_attention(inputs)
        else:
            outs = self._forward(inputs)
        return outs

    def _forward_fast_attention(self, inputs):
        """4 层 BiFPN 快速注意力模式"""
        p3_in, p4_in, p5_in, p6_in = inputs

        # ========== Top-Down ==========

        # P5_td ← P5_in + P6_in↑ (P6是最高层，不生成中间层，直接用于上采样)
        p5_w1 = self.p5_w1_relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        p6_up_interp = F.interpolate(p6_in, size=p5_in.shape[-2:], mode='nearest')
        p5_td = self.conv5up(self.swish(weight[0] * p5_in + weight[1] * p6_up_interp))

        # P4_td ← P4_in + P5_td↑
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        p5_up_interp = F.interpolate(p5_td, size=p4_in.shape[-2:], mode='nearest')
        p4_td = self.conv4up(self.swish(weight[0] * p4_in + weight[1] * p5_up_interp))

        # P3_td ← P3_in + P4_td↑
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        p4_up_interp = F.interpolate(p4_td, size=p3_in.shape[-2:], mode='nearest')
        p3_td = self.conv3up(self.swish(weight[0] * p3_in + weight[1] * p4_up_interp))

        # ========== Bottom-Up ==========

        # P3_out ← P3_td
        p3_out = p3_td

        # P4_out ← P4_in + P4_td + P3_out↓
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        p4_down = self.p4_downsample(p3_out)
        p4_out = self.conv4down(
            self.swish(weight[0] * p4_in + weight[1] * p4_td + weight[2] * p4_down)
        )

        # P5_out ← P5_in + P5_td + P4_out↓
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        p5_down = self.p5_downsample(p4_out)
        p5_out = self.conv5down(
            self.swish(weight[0] * p5_in + weight[1] * p5_td + weight[2] * p5_down)
        )

        # P6_out ← P6_in + P5_out↓ (最高层，只有2个输入，不使用P6_td)
        p6_w2 = self.p6_w2_relu(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        p6_down = self.p5_downsample(p5_out)
        p6_out = self.conv6down(
            self.swish(weight[0] * p6_in + weight[1] * p6_down)
        )

        return p3_out, p4_out, p5_out, p6_out

    def _forward(self, inputs):
        """4 层 BiFPN 标准模式（无权重归一化）"""
        p3_in, p4_in, p5_in, p6_in = inputs

        # ========== Top-Down ==========

        # P5_td ← P5_in + P6_in↑ (P6是最高层，不生成中间层，直接用于上采样)
        p6_up_interp = F.interpolate(p6_in, size=p5_in.shape[-2:], mode='nearest')
        p5_td = self.conv5up(self.swish(p5_in + p6_up_interp))

        # P4_td ← P4_in + P5_td↑
        p5_up_interp = F.interpolate(p5_td, size=p4_in.shape[-2:], mode='nearest')
        p4_td = self.conv4up(self.swish(p4_in + p5_up_interp))

        # P3_td ← P3_in + P4_td↑
        p4_up_interp = F.interpolate(p4_td, size=p3_in.shape[-2:], mode='nearest')
        p3_td = self.conv3up(self.swish(p3_in + p4_up_interp))

        # P3_out
        p3_out = p3_td

        # P4_out ← P4_in + P4_td + P3_out↓
        p4_down = self.p4_downsample(p3_out)
        p4_out = self.conv4down(self.swish(p4_in + p4_td + p4_down))

        # P5_out ← P5_in + P5_td + P4_out↓
        p5_down = self.p5_downsample(p4_out)
        p5_out = self.conv5down(self.swish(p5_in + p5_td + p5_down))

        # P6_out ← P6_in + P5_out↓ (最高层，只有2个输入，不使用P6_td)
        p6_down = self.p5_downsample(p5_out)
        p6_out = self.conv6down(self.swish(p6_in + p6_down))

        return p3_out, p4_out, p5_out, p6_out


class BiFPNUnified(nn.Module):
    """
    统一通道数的 BiFPN (4 层版本)

    处理 backbone 输出 [C3, C4, C5]，先从 C5 生成 C6，然后投影并送入 BiFPN
    """

    def __init__(self, backbone_channels=[512, 1024, 2048], unify_channels=256,
                 epsilon=1e-4, onnx_export=False, attention=True):
        """
        Args:
            backbone_channels: backbone 输出的通道数 [C3, C4, C5]
            unify_channels: 统一后的通道数
            epsilon: 权重归一化的小常数
            onnx_export: 是否导出 ONNX
            attention: 是否使用快速注意力模式
        """
        super(BiFPNUnified, self).__init__()

        self.backbone_channels = backbone_channels
        self.unify_channels = unify_channels

        # 输入投影层：将 C3, C4, C5, C6 投影到统一维度
        # C6 的通道数与 C5 相同 (都是 2048)
        self.input_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, unify_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(unify_channels, momentum=0.01, eps=1e-3),
                nn.SiLU()
            )
            for in_ch in backbone_channels + [backbone_channels[-1]]  # [512, 1024, 2048, 2048]
        ])

        # BiFPN 核心 (4 层)
        self.bifpn = BiFPN(
            num_channels=unify_channels,
            epsilon=epsilon,
            onnx_export=onnx_export,
            attention=attention
        )

    def forward(self, features, masks=None):
        """
        Args:
            features: list of tensors [C3, C4, C5] from backbone
            masks: list of tensors (未使用)

        Returns:
            enhanced_features: list of tensors [P3, P4, P5, P6]
        """
        c3, c4, c5 = features[0].tensors, features[1].tensors, features[2].tensors

        # 1. 从 C5 生成 C6 (在投影之前，与 Deformable DETR 官方一致)
        c6 = F.max_pool2d(c5, kernel_size=3, stride=2, padding=1)

        # 2. 投影到统一维度 [C3, C4, C5, C6] → [P3, P4, P5, P6]
        p3 = self.input_projs[0](c3)
        p4 = self.input_projs[1](c4)
        p5 = self.input_projs[2](c5)
        p6 = self.input_projs[3](c6)

        # 3. BiFPN 融合 (4 层输入 [P3, P4, P5, P6])
        enhanced_features = self.bifpn([p3, p4, p5, p6])

        return enhanced_features


def build_bifpn(channels, hidden_dim=256):
    """
    构建 BiFPN 模块

    Args:
        channels: backbone 输出的通道数 [C3, C4, C5]
        hidden_dim: Transformer 的隐藏维度

    Returns:
        BiFPN 模块 (4 层版本: P3, P4, P5, P6)
    """
    return BiFPNUnified(backbone_channels=channels, unify_channels=hidden_dim)
