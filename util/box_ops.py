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
from torchvision.ops.boxes import box_area


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
