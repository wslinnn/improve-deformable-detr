# ------------------------------------------------------------------------
# DN-DETR Components for Denoising Training
# 去噪训练组件
# 参考: https://github.com/IDEA-Research/DN-DETR
# 适配标准 Deformable DETR 架构 (非DAB版本)
# ------------------------------------------------------------------------

import torch
import torch.nn.functional as F
import math
from util.misc import accuracy, inverse_sigmoid
from util import box_ops


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Focal Loss for classification (与官方DN-DETR一致)
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def get_proposal_pos_embed(proposals, hidden_dim, num_pos_feats=128, temperature=10000):
    """
    将4维bbox编码到hidden_dim维，用于生成position embedding
    参考DeformableTransformer.get_proposal_pos_embed的实现

    Args:
        proposals: [N, 4] bbox in cxcywh format (已通过inverse_sigmoid编码)
        hidden_dim: 目标维度
        num_pos_feats: 每个坐标使用的特征维度
        temperature: 温度参数

    Returns:
        encoded: [N, hidden_dim] 编码后的position embedding
    """
    device = proposals.device
    scale = 2 * math.pi

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    # proposals: [N, 4], 先sigmoid再scale
    proposals = proposals.sigmoid() * scale
    # proposals: [N, 4, 128]
    pos = proposals[:, :, None] / dim_t
    # 使用sin和cos交替编码
    pos = torch.stack((pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3).flatten(2)
    # pos: [N, 4, 2*num_pos_feats] -> [N, 8*num_pos_feats]
    pos = pos.flatten(1)

    # 如果hidden_dim不是8的倍数，截断或填充
    if pos.shape[1] < hidden_dim:
        padding = torch.zeros(pos.shape[0], hidden_dim - pos.shape[1], device=device)
        pos = torch.cat([pos, padding], dim=1)
    elif pos.shape[1] > hidden_dim:
        pos = pos[:, :hidden_dim]

    return pos


def prepare_for_dn(dn_args, tgt_weight, embedweight, batch_size, training,
                   num_queries, num_classes, hidden_dim, label_enc, device='cuda'):
    """
    准备去噪训练的输入 - 适配标准Deformable DETR架构

    重要：标准Deformable DETR与DAB-DETR的query embedding格式不同：
    - DAB-DETR: content=[hidden_dim], position=[4]（直接bbox坐标）
    - 标准Deformable DETR: content=[hidden_dim], position=[hidden_dim]（position embedding）

    因此DN queries的position部分也需要编码为[hidden_dim]格式。

    Args:
        dn_args: (targets, scalar, label_noise_scale, box_noise_scale, num_patterns) 或 num_patterns
        tgt_weight: content部分 [num_queries, hidden_dim]
        embedweight: position部分 [num_queries, hidden_dim]
        batch_size: bs
        training: training mode
        num_queries: number of queries
        num_classes: number of classes
        hidden_dim: hidden dimension
        label_enc: label encoder
        device: device to use

    Returns:
        input_query_label: [batch_size, num_dn_queries + num_queries, hidden_dim]
        input_query_bbox: [batch_size, num_dn_queries + num_queries, hidden_dim]
        attn_mask: attention mask
        mask_dict: dict containing DN information
    """
    if training:
        targets, scalar, label_noise_scale, box_noise_scale, num_patterns = dn_args
    else:
        num_patterns = dn_args

    if num_patterns == 0:
        num_patterns = 1

    # 构造正常queries的content和position embedding
    # indicator0: 0表示正常query
    indicator0 = torch.zeros([num_queries * num_patterns, 1], device=device)
    # tgt: [num_queries, hidden_dim] - 实际content + indicator
    # 为了与DN queries保持一致，我们需要把tgt_weight的前hidden_dim-1维作为content，最后1维作为indicator
    tgt_content = tgt_weight[:, :hidden_dim-1]  # [num_queries, hidden_dim-1]
    tgt = torch.cat([tgt_content, indicator0], dim=1)  # [num_queries, hidden_dim]
    refpoint_emb = embedweight  # [num_queries, hidden_dim] - position embedding

    if training:
        known = [(torch.ones_like(t['labels'])).to(device) for t in targets]
        know_idx = [torch.nonzero(t) for t in known]
        known_num = [sum(k) for k in known]

        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t['labels'] for t in targets])
        boxes = torch.cat([t['boxes'] for t in targets])
        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)

        # add noise
        known_indice = known_indice.repeat(scalar, 1).view(-1)
        known_labels = labels.repeat(scalar, 1).view(-1)
        known_bid = batch_idx.repeat(scalar, 1).view(-1)
        known_bboxs = boxes.repeat(scalar, 1)
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        # noise on the label
        if label_noise_scale > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < (label_noise_scale)).view(-1)
            new_label = torch.randint_like(chosen_indice, 0, num_classes)
            known_labels_expaned.scatter_(0, chosen_indice, new_label)

        # noise on the box
        if box_noise_scale > 0:
            diff = torch.zeros_like(known_bbox_expand)
            diff[:, :2] = known_bbox_expand[:, 2:] / 2
            diff[:, 2:] = known_bbox_expand[:, 2:]
            known_bbox_expand += torch.mul((torch.rand_like(known_bbox_expand) * 2 - 1.0),
                                          diff).to(device) * box_noise_scale
            known_bbox_expand = known_bbox_expand.clamp(min=0.0, max=1.0)

        # encode labels
        m = known_labels_expaned.long().to(device)
        input_label_embed = label_enc(m)  # [N, hidden_dim-1]
        # add dn part indicator
        indicator1 = torch.ones([input_label_embed.shape[0], 1], device=device)
        input_label_embed = torch.cat([input_label_embed, indicator1], dim=1)  # [N, hidden_dim]

        # encode boxes to position embedding
        # 关键：标准Deformable DETR需要position embedding格式，不是直接的bbox坐标
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)  # [N, 4]
        input_bbox_embed = get_proposal_pos_embed(input_bbox_embed, hidden_dim)  # [N, hidden_dim]

        single_pad = int(max(known_num))
        pad_size = int(single_pad * scalar)
        padding_label = torch.zeros(pad_size, hidden_dim, device=device)
        padding_bbox = torch.zeros(pad_size, hidden_dim, device=device)
        input_query_label = torch.cat([padding_label, tgt], dim=0).repeat(batch_size, 1, 1)
        input_query_bbox = torch.cat([padding_bbox, refpoint_emb], dim=0).repeat(batch_size, 1, 1)

        # map in order
        map_known_indice = torch.tensor([], device=device)
        if len(known_num):
            map_known_indice = torch.cat([torch.tensor(range(num), device=device) for num in known_num])
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(scalar)]).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

        tgt_size = pad_size + num_queries * num_patterns
        attn_mask = torch.ones(tgt_size, tgt_size, device=device) < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(scalar):
            if i == 0:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            if i == scalar - 1:
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            else:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        mask_dict = {
            'known_indice': torch.as_tensor(known_indice).long(),
            'batch_idx': torch.as_tensor(batch_idx).long(),
            'map_known_indice': torch.as_tensor(map_known_indice).long(),
            'known_lbs_bboxes': (known_labels, known_bboxs),
            'know_idx': know_idx,
            'pad_size': pad_size
        }
    else:  # no dn for inference
        input_query_label = tgt.repeat(batch_size, 1, 1)
        input_query_bbox = refpoint_emb.repeat(batch_size, 1, 1)
        attn_mask = None
        mask_dict = None

    return input_query_label, input_query_bbox, attn_mask, mask_dict


def dn_post_process(outputs_class, outputs_coord, mask_dict):
    """
    post process of dn after output from the transformer
    put the dn part in the mask_dict
    """
    if mask_dict and mask_dict['pad_size'] > 0:
        output_known_class = outputs_class[:, :, :mask_dict['pad_size'], :]
        output_known_coord = outputs_coord[:, :, :mask_dict['pad_size'], :]
        outputs_class = outputs_class[:, :, mask_dict['pad_size']:, :]
        outputs_coord = outputs_coord[:, :, mask_dict['pad_size']:, :]
        mask_dict['output_known_lbs_bboxes'] = (output_known_class, output_known_coord)
    return outputs_class, outputs_coord


def prepare_for_loss(mask_dict):
    """
    prepare dn components to calculate loss
    """
    output_known_class, output_known_coord = mask_dict['output_known_lbs_bboxes']
    known_labels, known_bboxs = mask_dict['known_lbs_bboxes']
    map_known_indice = mask_dict['map_known_indice']

    known_indice = mask_dict['known_indice']

    batch_idx = mask_dict['batch_idx']
    bid = batch_idx[known_indice]
    if len(output_known_class) > 0:
        output_known_class = output_known_class.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
        output_known_coord = output_known_coord.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
    num_tgt = known_indice.numel()
    return known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt


def tgt_loss_boxes(src_boxes, tgt_boxes, num_tgt):
    """Compute the losses related to the bounding boxes"""
    if len(tgt_boxes) == 0:
        return {
            'tgt_loss_bbox': torch.as_tensor(0.).to(src_boxes.device),
            'tgt_loss_giou': torch.as_tensor(0.).to(src_boxes.device),
        }

    loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction='none')

    losses = {}
    losses['tgt_loss_bbox'] = loss_bbox.sum() / num_tgt

    loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
        box_ops.box_cxcywh_to_xyxy(src_boxes),
        box_ops.box_cxcywh_to_xyxy(tgt_boxes)))
    losses['tgt_loss_giou'] = loss_giou.sum() / num_tgt
    return losses


def tgt_loss_labels(src_logits_, tgt_labels_, num_tgt, focal_alpha, log=True):
    """Classification loss (NLL)"""
    if len(tgt_labels_) == 0:
        return {
            'tgt_loss_ce': torch.as_tensor(0.).to(src_logits_.device),
            'tgt_class_error': torch.as_tensor(0.).to(src_logits_.device),
        }

    src_logits, tgt_labels = src_logits_.unsqueeze(0), tgt_labels_.unsqueeze(0)

    target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                        dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
    target_classes_onehot.scatter_(2, tgt_labels.unsqueeze(-1), 1)

    target_classes_onehot = target_classes_onehot[:, :, :-1]
    loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_tgt, alpha=focal_alpha, gamma=2) * src_logits.shape[1]

    losses = {'tgt_loss_ce': loss_ce}

    losses['tgt_class_error'] = 100 - accuracy(src_logits_, tgt_labels_)[0]
    return losses


def compute_dn_loss(mask_dict, training, aux_num, focal_alpha):
    """
    compute dn loss in criterion
    """
    losses = {}
    if training and 'output_known_lbs_bboxes' in mask_dict:
        known_labels, known_bboxs, output_known_class, output_known_coord, \
        num_tgt = prepare_for_loss(mask_dict)
        losses.update(tgt_loss_labels(output_known_class[-1], known_labels, num_tgt, focal_alpha))
        losses.update(tgt_loss_boxes(output_known_coord[-1], known_bboxs, num_tgt))
    else:
        losses['tgt_loss_bbox'] = torch.as_tensor(0.).cuda()
        losses['tgt_loss_giou'] = torch.as_tensor(0.).cuda()
        losses['tgt_loss_ce'] = torch.as_tensor(0.).cuda()
        losses['tgt_class_error'] = torch.as_tensor(0.).cuda()

    if aux_num:
        for i in range(aux_num):
            # dn aux loss
            if training and 'output_known_lbs_bboxes' in mask_dict:
                l_dict = tgt_loss_labels(output_known_class[i], known_labels, num_tgt, focal_alpha)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
                l_dict = tgt_loss_boxes(output_known_coord[i], known_bboxs, num_tgt)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
            else:
                l_dict = dict()
                l_dict['tgt_loss_bbox'] = torch.as_tensor(0.).cuda()
                l_dict['tgt_class_error'] = torch.as_tensor(0.).cuda()
                l_dict['tgt_loss_giou'] = torch.as_tensor(0.).cuda()
                l_dict['tgt_loss_ce'] = torch.as_tensor(0.).cuda()
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
    return losses
