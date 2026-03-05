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
    3 层 BiFPN: 处理 P3, P4, P5

    与官方 EfficientDet BiFPN 逻辑一致，简化为 3 层版本

    数据流:
        P5_0 ---------> P5_1 ---------> P5_2
           |             |                ↑
           |             |                |
        P4_0 ---------> P4_1 ---------> P4_2
           |             |                ↑
           |             |                |
        P3_0 -----------------------> P3_2

    Top-Down:
        P4_1 = P4_0 + P5_0↑
        P3_2 = P3_0 + P4_1↑
        (P5_1 无上层，所以 P5_1 = P5_0)

    Bottom-Up:
        P4_2 = P4_0 + P4_1 + P3_2↓
        P5_2 = P5_0 + P5_1 + P4_2↓
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

        # ========== Top-Down 卷积层 ==========
        self.conv5up = SeparableConvBlock(num_channels, onnx_export=onnx_export)   # P5→P4
        self.conv4up = SeparableConvBlock(num_channels, onnx_export=onnx_export)   # P4→P3
        self.conv3up = SeparableConvBlock(num_channels, onnx_export=onnx_export)   # P3 输出

        # ========== Bottom-Up 卷积层 ==========
        self.conv4down = SeparableConvBlock(num_channels, onnx_export=onnx_export)  # P3→P4
        self.conv5down = SeparableConvBlock(num_channels, onnx_export=onnx_export)  # P4→P5

        # ========== 上采样层 ==========
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # ========== 下采样层 ==========
        self.p4_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p5_downsample = MaxPool2dStaticSamePadding(3, 2)

        # Swish 激活函数
        if not self.onnx_export:
            self.swish = MemoryEfficientSwish()
        else:
            self.swish = Swish()

        # ========== 可学习权重 (Top-Down) ==========
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1_relu = nn.ReLU()
        # P4 ← P4 + P5↑
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        # P3 ← P3 + P4↑
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()

        # ========== 可学习权重 (Bottom-Up) ==========
        # P4 ← P4 + P4_up + P3↓
        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        # P5 ← P5 + P5_up + P4↓
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()

    def forward(self, inputs):
        """
        Args:
            inputs: list of tensors [p3_in, p4_in, p5_in]

        Returns:
            outs: list of tensors [p3_out, p4_out, p5_out]
        """
        if self.attention:
            outs = self._forward_fast_attention(inputs)
        else:
            outs = self._forward(inputs)
        return outs

    def _forward_fast_attention(self, inputs):
        """快速注意力模式"""
        p3_in, p4_in, p5_in = inputs

        # ========== Top-Down ==========

        # P5_1: 最高层，无上层输入，P5_1 = P5_0 (经过 conv)
        p5_w1 = self.p5_w1_relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        p5_up = self.conv5up(self.swish(weight[0] * p5_in))

        # P4_1 ← P4_0 + P5_1↑ (动态上采样匹配尺寸)
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        p5_up_interp = F.interpolate(p5_up, size=p4_in.shape[-2:], mode='nearest')
        p4_up = self.conv4up(self.swish(weight[0] * p4_in + weight[1] * p5_up_interp))

        # P3_2 ← P3_0 + P4_1↑ (动态上采样匹配尺寸)
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        p4_up_interp = F.interpolate(p4_up, size=p3_in.shape[-2:], mode='nearest')
        p3_out = self.conv3up(self.swish(weight[0] * p3_in + weight[1] * p4_up_interp))

        # ========== Bottom-Up ==========

        # P4_2 ← P4_0 + P4_1 + P3_2↓
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        p4_out = self.conv4down(
            self.swish(weight[0] * p4_in + weight[1] * p4_up + weight[2] * self.p4_downsample(p3_out))
        )

        # P5_2 ← P5_0 + P5_1 + P4_2↓
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        p5_out = self.conv5down(
            self.swish(weight[0] * p5_in + weight[1] * p5_up + weight[2] * self.p5_downsample(p4_out))
        )

        return p3_out, p4_out, p5_out

    def _forward(self, inputs):
        """标准模式（无权重归一化）"""
        p3_in, p4_in, p5_in = inputs

        # ========== Top-Down ==========

        # P5_1: 最高层
        p5_up = self.conv5up(self.swish(p5_in))

        # P4_1 ← P4_0 + P5_1 (动态上采样匹配尺寸)
        p5_up_interp = F.interpolate(p5_up, size=p4_in.shape[-2:], mode='nearest')
        p4_up = self.conv4up(self.swish(p4_in + p5_up_interp))

        # P3_2 ← P3_0 + P4_1 (动态上采样匹配尺寸)
        p4_up_interp = F.interpolate(p4_up, size=p3_in.shape[-2:], mode='nearest')
        p3_out = self.conv3up(self.swish(p3_in + p4_up_interp))

        # ========== Bottom-Up ==========

        # P4_2 ← P4_0 + P4_1 + P3_2
        p4_out = self.conv4down(
            self.swish(p4_in + p4_up + self.p4_downsample(p3_out))
        )

        # P5_2 ← P5_0 + P5_1 + P4_2
        p5_out = self.conv5down(
            self.swish(p5_in + p5_up + self.p5_downsample(p4_out))
        )

        return p3_out, p4_out, p5_out


class BiFPNUnified(nn.Module):
    """
    统一通道数的 BiFPN

    处理 backbone 输出 [C3, C4, C5]，投影后送入 BiFPN
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

        # 输入投影层：将 C3, C4, C5 投影到统一维度
        self.input_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, unify_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(unify_channels, momentum=0.01, eps=1e-3),
                nn.SiLU()
            )
            for in_ch in backbone_channels
        ])

        # BiFPN 核心 (3 层)
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
            enhanced_features: list of tensors [P3, P4, P5]
        """
        # 投影到统一维度
        unified_features = []
        for i, feat in enumerate(features):
            unified = self.input_projs[i](feat)
            unified_features.append(unified)

        # BiFPN 融合 (3 层)
        enhanced_features = self.bifpn(unified_features)

        return enhanced_features


def build_bifpn(channels, hidden_dim=256):
    """
    构建 BiFPN 模块

    Args:
        channels: backbone 输出的通道数 [C3, C4, C5]
        hidden_dim: Transformer 的隐藏维度

    Returns:
        BiFPN 模块 (3 层版本)
    """
    return BiFPNUnified(backbone_channels=channels, unify_channels=hidden_dim)
