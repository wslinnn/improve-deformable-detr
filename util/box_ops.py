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


def box_to_gaussian(boxes, eps=1e-7):
    """
    将边界框建模为二维高斯分布

    假设边界框内服从均匀分布，则方差 = (range)² / 12

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

    # 数值稳定性：防止 w, h 为 0 或负数
    w = w.clamp(min=eps)
    h = h.clamp(min=eps)

    # 均值是边界框的中心点
    mu = torch.stack([cx, cy], dim=-1)

    # 对于均匀分布，方差 = range² / 12，标准差 = range / sqrt(12)
    # 使用预计算的 SQRT_12 常数，避免重复创建 tensor
    sigma_w = w / SQRT_12
    sigma_h = h / SQRT_12

    sigma = torch.stack([sigma_w, sigma_h], dim=-1)

    return mu, sigma


def nwd_similarity(boxes1, boxes2, eps=1e-7):
    """
    计算两个框集合之间的 NWD 相似度矩阵

    注意：返回值是相似度 (1 表示相同，0 表示不同)，不是距离
    如果用于 Loss 计算且框是一一对应的，请使用 nwd_loss 以获得更高性能

    Args:
        boxes1: (N, 4) in cxcywh format
        boxes2: (M, 4) in cxcywh format
        eps: 小常数，用于数值稳定性

    Returns:
        nwd: (N, M) NWD 相似度矩阵，值域 (0, 1]
              1 表示完全相同，0 表示完全不相关
    """
    mu1, sigma1 = box_to_gaussian(boxes1, eps)
    mu2, sigma2 = box_to_gaussian(boxes2, eps)

    # (N, 1, 2) - (1, M, 2) -> (N, M, 2)
    center_dist_sq = (mu1[:, None, :] - mu2[None, :, :]).pow(2).sum(-1)
    sigma_dist_sq = (sigma1[:, None, :] - sigma2[None, :, :]).pow(2).sum(-1)

    wasserstein_sq = center_dist_sq + sigma_dist_sq

    # NWD 相似度：值域 (0, 1]
    similarity = torch.exp(-wasserstein_sq / 2.0)
    return similarity

def nwd_loss(boxes_pred, boxes_target, eps=1e-7):
    """
    基于 NWD 的损失函数 (一对一匹配模式)

    优化版：直接计算对角线元素，避免 O(N²) 的矩阵计算

    Args:
        boxes_pred: (N, 4) 预测框，cxcywh 格式
        boxes_target: (N, 4) 目标框，cxcywh 格式
        eps: 小常数，用于数值稳定性

    Returns:
        loss: NWD 损失，标量
    """
    # 1. 转换为高斯参数
    mu1, sigma1 = box_to_gaussian(boxes_pred, eps)
    mu2, sigma2 = box_to_gaussian(boxes_target, eps)

    # 2. 直接计算对应元素的距离 (Element-wise)，不广播
    # 形状均为 (N, 2)
    center_dist_sq = (mu1 - mu2).pow(2).sum(dim=-1)
    sigma_dist_sq = (sigma1 - sigma2).pow(2).sum(dim=-1)

    wasserstein_sq = center_dist_sq + sigma_dist_sq

    # 3. 计算相似度
    nwd_sim = torch.exp(-wasserstein_sq / 2.0)

    # 4. 计算损失：1 - similarity
    # 如果 nwd_sim 接近 1 (框重合)，loss 接近 0
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
