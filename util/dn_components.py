# ------------------------------------------------------------------------
# DN-DETR Components for Denoising Training
# 去噪训练组件
# 完全参考DINO实现: https://github.com/IDEA-Research/DN-DETR
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


def prepare_for_cdn(dn_args, training, num_queries, num_classes, hidden_dim, label_enc):
    """
    完全参考DINO实现的CDN (Conditional Denoising) 去噪训练
    支持两阶段模式

    参数格式 (dn_args):
        training: (targets, dn_number, label_noise_ratio, box_noise_scale)
        inference: num_queries

    Args:
        dn_args: 训练时为(targets, dn_number, label_noise_ratio, box_noise_scale)
        training: 是否为训练模式
        num_queries: query数量
        num_classes: 类别数
        hidden_dim: 隐藏层维度
        label_enc: 标签编码器

    Returns:
        input_query_label: [batch_size, num_dn_queries, hidden_dim]
        input_query_bbox: [batch_size, num_dn_queries, 4]
        attn_mask: attention mask
        dn_meta: 包含dn信息的字典
    """
    if training:
        targets, dn_number, label_noise_ratio, box_noise_scale = dn_args
        # positive and negative dn queries
        dn_number = dn_number * 2
        # 获取设备信息 - 从label_enc的权重获取，确保与模型设备一致
        device = label_enc.weight.device
        known = [(torch.ones_like(t['labels'])).to(device) for t in targets]
        batch_size = len(known)
        known_num = [sum(k) for k in known]

        if int(max(known_num)) == 0:
            dn_number = 1
        else:
            if dn_number >= 100:
                dn_number = dn_number // (int(max(known_num) * 2))
            elif dn_number < 1:
                dn_number = 1
        if dn_number == 0:
            dn_number = 1

        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t['labels'].to(device) for t in targets])
        boxes = torch.cat([t['boxes'].to(device) for t in targets])
        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i).to(device) for i, t in enumerate(targets)])

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)

        # 扩展为正负样本
        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
        known_bboxs = boxes.repeat(2 * dn_number, 1)
        known_labels_expanded = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        # label noise (与DINO一致)
        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expanded.float())
            chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(-1)  # half of bbox prob
            new_label = torch.randint_like(chosen_indice, 0, num_classes)  # randomly put a new one here
            known_labels_expanded.scatter_(0, chosen_indice, new_label)

        single_pad = int(max(known_num))
        pad_size = int(single_pad * 2 * dn_number)

        # 计算positive和negative的索引 (与DINO一致)
        positive_idx = torch.tensor(range(len(boxes))).long().to(device).unsqueeze(0).repeat(dn_number, 1)
        positive_idx += (torch.tensor(range(dn_number)) * len(boxes) * 2).long().to(device).unsqueeze(1)
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(boxes)

        # box noise (完全按照DINO实现：在xyxy空间加噪声)
        if box_noise_scale > 0:
            # 转换到xyxy格式
            known_bbox_ = torch.zeros_like(known_bboxs)
            known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
            known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2

            diff = torch.zeros_like(known_bboxs)
            diff[:, :2] = known_bboxs[:, 2:] / 2
            diff[:, 2:] = known_bboxs[:, 2:] / 2

            # 对positive和negative使用不同的随机处理
            rand_sign = torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
            rand_part = torch.rand_like(known_bboxs)
            rand_part[negative_idx] += 1.0  # negative噪声更大
            rand_part *= rand_sign

            known_bbox_ = known_bbox_ + torch.mul(rand_part, diff).to(device) * box_noise_scale
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)

            # 转换回cxcywh格式
            known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]

        # 编码labels (适配标准Deformable DETR的label_enc维度为hidden_dim-1)
        m = known_labels_expanded.long().to(device)
        input_label_embed = label_enc(m)
        # 填充到hidden_dim维（标准Deformable DETR的label_enc是hidden_dim-1维）
        if input_label_embed.shape[-1] < hidden_dim:
            padding = torch.zeros(*input_label_embed.shape[:-1], hidden_dim - input_label_embed.shape[-1], device=input_label_embed.device)
            input_label_embed = torch.cat([input_label_embed, padding], dim=-1)

        # 将bbox转换为position embedding（与标准Deformable DETR一致）
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)
        input_bbox_embed = get_proposal_pos_embed(input_bbox_embed, hidden_dim)

        # padding (转换为hidden_dim维以兼容标准Deformable DETR)
        padding_label = torch.zeros(pad_size, hidden_dim, device=device)
        padding_bbox = torch.zeros(pad_size, hidden_dim, device=device)  # 修改为hidden_dim维

        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

        # 填充DN queries
        map_known_indice = torch.tensor([], device=device)
        if len(known_num):
            map_known_indice = torch.cat([torch.tensor(range(num), device=device) for num in known_num])
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(2 * dn_number)]).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

        # attention mask (与DINO一致)
        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size, device=device) < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(dn_number):
            if i == 0:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
            if i == dn_number - 1:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * i * 2] = True
            else:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * 2 * i] = True

        dn_meta = {
            'pad_size': pad_size,
            'num_dn_group': dn_number,
            'known_num': known_num,
            # 保存带噪声的bbox坐标（4维cxcywh格式，inverse_sigmoid后），用于transformer中的reference_points
            'dn_bbox_coords': inverse_sigmoid(known_bbox_expand),
        }
    else:
        # inference模式
        input_query_label = None
        input_query_bbox = None
        attn_mask = None
        dn_meta = None

    return input_query_label, input_query_bbox, attn_mask, dn_meta


# 保持原有函数名以兼容现有代码
def prepare_for_dn(dn_args, tgt_weight, embedweight, batch_size, training,
                   num_queries, num_classes, hidden_dim, label_enc, device='cuda'):
    """
    准备去噪训练的输入 - 适配标准Deformable DETR架构（支持两阶段模式）
    完全参考DINO的prepare_for_cdn实现

    Args:
        dn_args: (targets, scalar, label_noise_scale, box_noise_scale, num_patterns) 或 num_patterns
        tgt_weight: content部分 [num_queries, hidden_dim]，两阶段时为None
        embedweight: position部分 [num_queries, hidden_dim]，两阶段时为None
        batch_size: bs
        training: training mode
        num_queries: number of queries
        num_classes: number of classes
        hidden_dim: hidden dimension
        label_enc: label encoder
        device: device to use

    Returns:
        input_query_label: [batch_size, num_dn_queries, hidden_dim]
        input_query_bbox: [batch_size, num_dn_queries, 4]
        attn_mask: attention mask
        mask_dict: dict containing DN information (兼容现有代码)
    """
    if training:
        # 解析dn_args - 支持两种格式
        # 格式1: (targets, scalar, label_noise_scale, box_noise_scale, num_patterns)
        # 格式2: (targets, dn_number, label_noise_scale, box_noise_scale)
        if len(dn_args) >= 4:
            if len(dn_args) == 5:
                targets, scalar, label_noise_scale, box_noise_scale, num_patterns = dn_args
            else:
                targets, scalar, label_noise_scale, box_noise_scale = dn_args
                num_patterns = 1
        else:
            targets, scalar = dn_args[0], 1
            label_noise_scale, box_noise_scale = 0.0, 0.0
            num_patterns = 1

        # 转换为DINO格式: (targets, dn_number, label_noise_ratio, box_noise_scale)
        dn_args_dino = (targets, scalar, label_noise_scale, box_noise_scale)

        # 调用DINO风格的prepare_for_cdn (device参数由prepare_for_cdn内部根据数据自动获取)
        input_query_label, input_query_bbox, attn_mask, dn_meta = prepare_for_cdn(
            dn_args_dino, training, num_queries, num_classes, hidden_dim, label_enc
        )

        # prepare_for_cdn 已经从 label_enc.weight.device 获取设备，不需要额外处理

        # 获取带噪声的bbox坐标（4维，inverse_sigmoid后），用于transformer的reference_points
        dn_bbox_coords = dn_meta['dn_bbox_coords'] if dn_meta else None

        # 计算 single_pad（与DINO一致，用于loss计算）
        # known_num 在 dn_meta 中
        known_num = dn_meta.get('known_num', []) if dn_meta else []
        single_pad = int(max(known_num)) if known_num else 0
        scalar = dn_meta['num_dn_group'] // 2 if dn_meta else 0  # dn_number

        # 获取设备信息用于known_lbs_bboxes
        dn_device = dn_bbox_coords.device if dn_bbox_coords is not None else device

        mask_dict = {
            'pad_size': dn_meta['pad_size'] if dn_meta else 0,
            'num_dn_group': dn_meta['num_dn_group'] if dn_meta else 0,
            'known_lbs_bboxes': (torch.cat([t['labels'].to(dn_device) for t in targets]),
                                 torch.cat([t['boxes'].to(dn_device) for t in targets])) if targets else (None, None),
            'dn_bbox_coords': dn_bbox_coords,  # 带噪声的bbox坐标（4维）
            'single_pad': single_pad,  # 每个样本的GT数量
            'scalar': scalar,  # dn_number
            'known_num': known_num,  # 每个样本的GT数量列表
        }
    else:
        # inference模式
        input_query_label = None
        input_query_bbox = None
        attn_mask = None
        mask_dict = None
        dn_bbox_coords = None

    return input_query_label, input_query_bbox, attn_mask, mask_dict, dn_bbox_coords


def dn_post_process(outputs_class, outputs_coord, mask_dict):
    """
    post process of dn after output from the transformer
    put the dn part in the mask_dict
    """
    if mask_dict and mask_dict.get('pad_size', 0) > 0:
        pad_size = mask_dict['pad_size']
        output_known_class = outputs_class[:, :, :pad_size, :]
        output_known_coord = outputs_coord[:, :, :pad_size, :]
        outputs_class = outputs_class[:, :, pad_size:, :]
        outputs_coord = outputs_coord[:, :, pad_size:, :]
        mask_dict['output_known_lbs_bboxes'] = (output_known_class, output_known_coord)
    return outputs_class, outputs_coord


def prepare_for_loss(mask_dict):
    """
    prepare dn components to calculate loss
    """
    if mask_dict is None or 'output_known_lbs_bboxes' not in mask_dict:
        return None, None, None, None, 0

    output_known_class, output_known_coord = mask_dict['output_known_lbs_bboxes']
    known_labels, known_bboxs = mask_dict['known_lbs_bboxes']

    if known_labels is None or len(known_labels) == 0:
        return known_labels, known_bboxs, output_known_class, output_known_coord, 0

    map_known_indice = mask_dict.get('map_known_indice', None)
    known_indice = mask_dict.get('known_indice', None)
    batch_idx = mask_dict.get('batch_idx', None)

    if known_indice is not None and batch_idx is not None:
        bid = batch_idx[known_indice]
        if len(output_known_class) > 0 and map_known_indice is not None:
            output_known_class = output_known_class.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
            output_known_coord = output_known_coord.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)

    num_tgt = known_indice.numel() if known_indice is not None else len(known_labels)
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
    参照detrex/dino实现：使用known_num进行逐样本处理
    """
    losses = {}
    # 获取设备信息
    if mask_dict is not None and 'output_known_lbs_bboxes' in mask_dict:
        output_known_class, _ = mask_dict['output_known_lbs_bboxes']
        device = output_known_class.device
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if training and mask_dict is not None and 'output_known_lbs_bboxes' in mask_dict:
        output_known_class, output_known_coord = mask_dict['output_known_lbs_bboxes']
        known_labels, known_bboxs = mask_dict['known_lbs_bboxes']

        pad_size = mask_dict.get('pad_size', 0)
        batch_size = output_known_class.shape[1]  # [num_layers, batch, pad_size, ...]
        scalar = mask_dict.get('scalar', 0)  # dn_number
        single_pad = mask_dict.get('single_pad', 0)  # max(known_num)
        known_num = mask_dict.get('known_num', [])  # 每个样本的GT数量列表

        if known_labels is not None and len(known_labels) > 0 and pad_size > 0 and batch_size > 0 and scalar > 0 and len(known_num) == batch_size:
            # output_known_class: [num_layers, batch, pad_size, num_classes]
            # output_known_coord: [num_layers, batch, pad_size, 4]
            # known_labels: [total_gt] (所有batch的GT合并)
            # known_bboxs: [total_gt, 4]
            # known_num: [batch_size] 每个样本的GT数量

            output_class = output_known_class[-1]  # [batch, pad_size, num_classes]
            output_coord = output_known_coord[-1]   # [batch, pad_size, 4]

            total_loss_ce = 0
            total_loss_bbox = 0
            total_loss_giou = 0
            total_num_tgt = 0

            # 逐样本处理（与detrex/dino一致）
            # 计算累积偏移（known_labels/known_bboxs是按batch顺序拼接的）
            cumsum_known_num = [0]
            for num in known_num:
                cumsum_known_num.append(cumsum_known_num[-1] + num)

            for b in range(batch_size):
                num_gt_b = known_num[b]  # 当前样本的GT数量

                if num_gt_b > 0 and single_pad >= num_gt_b:
                    # 为当前batch构建索引（与detrex/dino一致）
                    # 正样本: [0,1,...,num_gt_b-1, single_pad, ..., single_pad+num_gt_b-1, ...]
                    t = torch.arange(num_gt_b, dtype=torch.long, device=output_class.device)
                    t = t.unsqueeze(0).repeat(scalar, 1)  # [scalar, num_gt_b]
                    tgt_idx = t.flatten()  # [scalar * num_gt_b]

                    output_idx = (torch.arange(scalar, dtype=torch.long, device=output_class.device) * single_pad).unsqueeze(1) + t
                    output_idx = output_idx.flatten()  # [scalar * num_gt_b]

                    # 提取对应的output
                    b_output_class = output_class[b, output_idx, :]  # [scalar * num_gt_b, num_classes]
                    b_output_coord = output_coord[b, output_idx, :]   # [scalar * num_gt_b, 4]

                    # 获取当前样本的labels和bbox（使用累积偏移）
                    offset = cumsum_known_num[b]
                    b_labels = known_labels[offset:offset + num_gt_b].repeat(scalar)
                    b_bboxs = known_bboxs[offset:offset + num_gt_b].repeat(scalar, 1)

                    if len(b_labels) > 0:
                        l_dict = tgt_loss_labels(b_output_class, b_labels, len(b_labels), focal_alpha)
                        total_loss_ce += l_dict['tgt_loss_ce']

                        l_dict = tgt_loss_boxes(b_output_coord, b_bboxs, len(b_labels))
                        total_loss_bbox += l_dict['tgt_loss_bbox']
                        total_loss_giou += l_dict['tgt_loss_giou']
                        total_num_tgt += len(b_labels)

            if total_num_tgt > 0:
                losses['tgt_loss_ce'] = total_loss_ce / batch_size
                losses['tgt_loss_bbox'] = total_loss_bbox / total_num_tgt
                losses['tgt_loss_giou'] = total_loss_giou / total_num_tgt
                losses['tgt_class_error'] = torch.as_tensor(0.).to(device)
            else:
                losses['tgt_loss_bbox'] = torch.as_tensor(0.).to(device)
                losses['tgt_loss_giou'] = torch.as_tensor(0.).to(device)
                losses['tgt_loss_ce'] = torch.as_tensor(0.).to(device)
                losses['tgt_class_error'] = torch.as_tensor(0.).to(device)
        else:
            losses['tgt_loss_bbox'] = torch.as_tensor(0.).to(device)
            losses['tgt_loss_giou'] = torch.as_tensor(0.).to(device)
            losses['tgt_loss_ce'] = torch.as_tensor(0.).to(device)
            losses['tgt_class_error'] = torch.as_tensor(0.).to(device)
    else:
        losses['tgt_loss_bbox'] = torch.as_tensor(0.).to(device)
        losses['tgt_loss_giou'] = torch.as_tensor(0.).to(device)
        losses['tgt_loss_ce'] = torch.as_tensor(0.).to(device)
        losses['tgt_class_error'] = torch.as_tensor(0.).to(device)

    # aux loss
    if aux_num:
        for i in range(aux_num):
            if training and mask_dict is not None and 'output_known_lbs_bboxes' in mask_dict:
                output_known_class, output_known_coord = mask_dict['output_known_lbs_bboxes']
                known_labels, known_bboxs = mask_dict['known_lbs_bboxes']
                batch_size = output_known_class.shape[1]
                scalar = mask_dict.get('scalar', 0)
                single_pad = mask_dict.get('single_pad', 0)
                known_num = mask_dict.get('known_num', [])

                if known_labels is not None and len(known_labels) > 0 and len(known_num) == batch_size:
                    output_class = output_known_class[i]
                    output_coord = output_known_coord[i]

                    # 计算累积偏移（known_labels/known_bboxs是按batch顺序拼接的）
                    cumsum_known_num = [0]
                    for num in known_num:
                        cumsum_known_num.append(cumsum_known_num[-1] + num)

                    total_loss_ce = 0
                    total_loss_bbox = 0
                    total_loss_giou = 0
                    total_num_tgt = 0

                    for b in range(batch_size):
                        num_gt_b = known_num[b]

                        if num_gt_b > 0 and single_pad >= num_gt_b:
                            t = torch.arange(num_gt_b, dtype=torch.long, device=output_class.device)
                            t = t.unsqueeze(0).repeat(scalar, 1)
                            tgt_idx = t.flatten()

                            output_idx = (torch.arange(scalar, dtype=torch.long, device=output_class.device) * single_pad).unsqueeze(1) + t
                            output_idx = output_idx.flatten()

                            b_output_class = output_class[b, output_idx, :]
                            b_output_coord = output_coord[b, output_idx, :]

                            # 获取当前样本的labels和bbox（使用累积偏移）
                            offset = cumsum_known_num[b]
                            b_labels = known_labels[offset:offset + num_gt_b].repeat(scalar)
                            b_bboxs = known_bboxs[offset:offset + num_gt_b].repeat(scalar, 1)

                            if len(b_labels) > 0:
                                l_dict = tgt_loss_labels(b_output_class, b_labels, len(b_labels), focal_alpha)
                                total_loss_ce += l_dict['tgt_loss_ce']
                                l_dict = tgt_loss_boxes(b_output_coord, b_bboxs, len(b_labels))
                                total_loss_bbox += l_dict['tgt_loss_bbox']
                                total_loss_giou += l_dict['tgt_loss_giou']
                                total_num_tgt += len(b_labels)

                    if total_num_tgt > 0:
                        losses[f'tgt_loss_ce_{i}'] = total_loss_ce / batch_size
                        losses[f'tgt_loss_bbox_{i}'] = total_loss_bbox / total_num_tgt
                        losses[f'tgt_loss_giou_{i}'] = total_loss_giou / total_num_tgt
                    else:
                        losses[f'tgt_loss_bbox_{i}'] = torch.as_tensor(0.).to(device)
                        losses[f'tgt_loss_giou_{i}'] = torch.as_tensor(0.).to(device)
                        losses[f'tgt_loss_ce_{i}'] = torch.as_tensor(0.).to(device)
                else:
                    losses[f'tgt_loss_bbox_{i}'] = torch.as_tensor(0.).to(device)
                    losses[f'tgt_loss_giou_{i}'] = torch.as_tensor(0.).to(device)
                    losses[f'tgt_loss_ce_{i}'] = torch.as_tensor(0.).to(device)
            else:
                losses[f'tgt_loss_bbox_{i}'] = torch.as_tensor(0.).to(device)
                losses[f'tgt_loss_giou_{i}'] = torch.as_tensor(0.).to(device)
                losses[f'tgt_loss_ce_{i}'] = torch.as_tensor(0.).to(device)
    return losses

