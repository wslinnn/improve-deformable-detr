# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
import math
from torchvision.ops.boxes import box_area

# NWD 常量：预计算 sqrt(12)，避免重复创建 tensor
SQRT_12 = math.sqrt(12.0)


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def complete_box_iou(boxes1, boxes2, eps=1e-7):
    """
    Complete IoU (CIoU) from https://arxiv.org/abs/1911.08287

    CIoU = IoU - (center_distance / diagonal_length) - aspect_ratio_consistency

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)

    Args:
        boxes1: predicted boxes in xyxy format, shape [N, 4]
        boxes2: ground truth boxes in xyxy format, shape [M, 4]
        eps: small value for numerical stability
    """
    # degenerate boxes gives inf / nan results
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    # Compute IoU
    iou, union = box_iou(boxes1, boxes2)

    # Get center points
    center_x1 = (boxes1[:, 0] + boxes1[:, 2]) / 2
    center_y1 = (boxes1[:, 1] + boxes1[:, 3]) / 2
    center_x2 = (boxes2[:, 0] + boxes2[:, 2]) / 2
    center_y2 = (boxes2[:, 1] + boxes2[:, 3]) / 2

    # Compute widths and heights
    w1 = boxes1[:, 2] - boxes1[:, 0]
    h1 = boxes1[:, 3] - boxes1[:, 1]
    w2 = boxes2[:, 2] - boxes2[:, 0]
    h2 = boxes2[:, 3] - boxes2[:, 1]

    # Center distance squared (normalized by diagonal)
    center_dist_sq = (center_x1[:, None] - center_x2) ** 2 + (center_y1[:, None] - center_y2) ** 2

    # Find smallest enclosing box
    x1_enclosing = torch.min(boxes1[:, None, 0], boxes2[:, 0])
    y1_enclosing = torch.min(boxes1[:, None, 1], boxes2[:, 1])
    x2_enclosing = torch.max(boxes1[:, None, 2], boxes2[:, 2])
    y2_enclosing = torch.max(boxes1[:, None, 3], boxes2[:, 3])

    # Diagonal length squared of enclosing box
    diagonal_sq = (x2_enclosing - x1_enclosing) ** 2 + (y2_enclosing - y1_enclosing) ** 2

    # Aspect ratio consistency
    # v = (4 / pi^2) * (arctan(w_gt / h_gt) - arctan(w_pred / h_pred))^2
    # 使用 torch.pi 而非 math.pi，保持张量运算一致性
    ratio_pred = w1 / (h1 + eps)  # [N] for predicted boxes
    ratio_gt = w2 / (h2 + eps)    # [M] for ground truth boxes
    atan_pred = torch.atan(ratio_pred)
    atan_gt = torch.atan(ratio_gt)
    v = (4 / (torch.pi ** 2)) * (atan_pred[:, None] - atan_gt) ** 2  # [N, M]

    # Alpha weight parameter
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    # CIoU = IoU - (ρ²(b,b_gt) / c²) - αν
    # 注意：center_dist_sq 和 diagonal_sq 使用时都需要加 eps 防止除零
    ciou = iou - (center_dist_sq / (diagonal_sq + eps)) - alpha * v

    # clamp 保证数值稳定，防止极端值
    ciou = ciou.clamp(min=-1.0, max=1.0)

    return ciou


def nwd_similarity_single(pred, target, eps=1e-7, constant=2.0, wh_divisor=4):
    """
    计算一对一框的 NWD 相似度（cxcywh 格式，归一化坐标）

    参考: https://arxiv.org/abs/2005.03572

    Args:
        pred: (N, 4) 预测框，cxcywh 格式，归一化坐标 [0,1]
        target: (N, 4) 目标框，cxcywh 格式，归一化坐标 [0,1]
        eps: 数值稳定性常数
        constant: 缩放因子，归一化坐标建议 1.0~4.0，像素小目标用 8.0~12.8
        wh_divisor: 宽高距离除数，4=工程近似，12=理论公式

    Returns:
        similarity: (N,) 值域 (0,1]，1=完全相同
    """
    pred_cx, pred_cy, pred_w, pred_h = pred.unbind(-1)
    tgt_cx, tgt_cy, tgt_w, tgt_h = target.unbind(-1)

    # 确保宽高为正
    pred_w = pred_w + eps
    pred_h = pred_h + eps
    tgt_w = tgt_w + eps
    tgt_h = tgt_h + eps

    # 中心距离平方
    center_distance = (pred_cx - tgt_cx) ** 2 + (pred_cy - tgt_cy) ** 2 + eps

    # 尺寸距离（可切换 /4 或 /12）
    wh_distance = ((pred_w - tgt_w) ** 2 + (pred_h - tgt_h) ** 2) / wh_divisor

    wasserstein_2 = center_distance + wh_distance

    # NWD 相似度
    similarity = torch.exp(-torch.sqrt(wasserstein_2) / constant)
    return similarity


def box_to_gaussian(boxes, eps=1e-7):
    """
    将边界框建模为二维高斯分布

    假设边界框内服从均匀分布，则方差 = (range)² / 12

    注意：输入应该是归一化坐标 (0-1 范围)

    Args:
        boxes: (N, 4) in cxcywh format, each box is (cx, cy, w, h)
        eps: 防止除零或标准差为 0 的数值稳定性参数

    Returns:
        mu: (N, 2) 高斯分布的均值 (中心点坐标)
        sigma: (N, 2) 高斯分布的标准差 (std_w, std_h)
    """
    # 确保 boxes 是 float 类型
    boxes = boxes.float()
    cx, cy, w, h = boxes.unbind(-1)

    # 只对 w, h 做最小值限制，防止负数或零
    # 不对 cx, cy, w, h 做最大值限制，保持原始值
    w = torch.clamp(w, min=eps)
    h = torch.clamp(h, min=eps)

    # 均值是边界框的中心点
    mu = torch.stack([cx, cy], dim=-1)

    # 对于均匀分布，方差 = range² / 12，标准差 = range / sqrt(12)
    sigma_w = w / SQRT_12
    sigma_h = h / SQRT_12

    sigma = torch.stack([sigma_w, sigma_h], dim=-1)

    return mu, sigma


def nwd_similarity(boxes1, boxes2, eps=1e-7, constant=2.0, wh_divisor=4):
    """
    计算 N×M 的 NWD 相似度矩阵（用于匈牙利匹配）

    输入应该是归一化坐标 (0-1 范围)

    Args:
        boxes1: (N, 4) in cxcywh format, 归一化坐标 [0,1]
        boxes2: (M, 4) in cxcywh format, 归一化坐标 [0,1]
        eps: 数值稳定性常数
        constant: 缩放因子，归一化坐标建议 1.0~4.0
        wh_divisor: 宽高距离除数，4=工程近似，12=理论公式

    Returns:
        similarity: (N, M) NWD 相似度矩阵，值域 (0, 1]
    """
    b1_cx, b1_cy, b1_w, b1_h = boxes1.unbind(-1)
    b2_cx, b2_cy, b2_w, b2_h = boxes2.unbind(-1)

    b1_w = b1_w + eps
    b1_h = b1_h + eps
    b2_w = b2_w + eps
    b2_h = b2_h + eps

    # 广播计算 (N, M)
    center_distance = (b1_cx[:, None] - b2_cx[None, :]) ** 2 + \
                      (b1_cy[:, None] - b2_cy[None, :]) ** 2 + eps

    wh_distance = ((b1_w[:, None] - b2_w[None, :]) ** 2 +
                   (b1_h[:, None] - b2_h[None, :]) ** 2) / wh_divisor

    wasserstein_2 = center_distance + wh_distance
    similarity = torch.exp(-torch.sqrt(wasserstein_2) / constant)
    return similarity

def nwd_loss(boxes_pred, boxes_target, eps=1e-7, constant=2.0, wh_divisor=4):
    """
    NWD Loss (一对一匹配，cxcywh 归一化坐标)

    Args:
        boxes_pred: (N, 4) 预测框，cxcywh 格式，归一化坐标 [0,1]
        boxes_target: (N, 4) 目标框，cxcywh 格式，归一化坐标 [0,1]
        eps: 数值稳定性常数
        constant: 缩放因子，归一化坐标建议 1.0~4.0
        wh_divisor: 宽高距离除数，4=工程近似，12=理论公式

    Returns:
        loss: 标量，值域 [0, 1)
    """
    # 检查输入有效性（保持梯度）
    if boxes_pred.numel() == 0 or boxes_target.numel() == 0:
        return torch.tensor(0.0, device=boxes_pred.device, requires_grad=True)

    # 计算相似度 -> 转换为 loss
    nwd_sim = nwd_similarity_single(boxes_pred, boxes_target, eps, constant, wh_divisor)
    loss = (1.0 - nwd_sim).mean()
    return loss


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
