import torch
import torch.nn.functional as F
from det_oprs.bbox_opr import bbox_transform_inv_opr, bbox_transform_opr, box_overlap_opr, align_box_giou_opr
from config import config

EPS = 1e-6

def smooth_l1_loss(pred, target, beta: float):
    if beta < 1e-5:
        loss = torch.abs(input - target)
    else:
        abs_x = torch.abs(pred- target)
        in_mask = abs_x < beta
        loss = torch.where(in_mask, 0.5 * abs_x ** 2 / beta, abs_x - 0.5 * beta)
    return loss.sum(axis=-1)

def giou_loss(pred, target, anchor):
    pred_boxes = bbox_transform_inv_opr(anchor, pred)
    target_boxes = bbox_transform_inv_opr(anchor, target)
    gious = align_box_giou_opr(pred_boxes, target_boxes)
    loss = 1 - gious
    return loss

def focal_loss(inputs, targets, alpha=-1, gamma=2):
    class_range = torch.arange(1, inputs.shape[1] + 1, device=inputs.device)
    pos_pred = (1 - inputs) ** gamma * torch.log(inputs)
    neg_pred = inputs ** gamma * torch.log(1 - inputs)

    pos_loss = (targets == class_range) * pos_pred * alpha
    neg_loss = (targets != class_range) * neg_pred * (1 - alpha)
    loss = -(pos_loss + neg_loss)
    return loss.sum(axis=1)

def freeanchor_loss(anchors, cls_prob, bbox_preds, gt_boxes, im_info):
    gt_labels, gt_bboxes = [], []
    cls_prob = cls_prob.reshape(config.train_batch_per_gpu, -1, config.num_classes-1)
    bbox_preds = bbox_preds.reshape(config.train_batch_per_gpu, -1, 4)
    gt_boxes = [gt_boxes[bid, :int(im_info[bid, 5]), :] for bid in range(config.train_batch_per_gpu)]
    for gt_box in gt_boxes:
        obj_mask = torch.where(gt_box[:, -1] == 1)[0] 
        gt_labels.append(torch.zeros_like(gt_box[obj_mask, -1]).long())
        gt_bboxes.append(gt_box[obj_mask, :4])
    box_prob = []
    num_pos = 0
    positive_losses = []
    for _, (gt_labels_, gt_bboxes_, cls_prob_, bbox_preds_) in \
                        enumerate(zip(gt_labels, gt_bboxes, cls_prob, bbox_preds)):
        with torch.no_grad():
            if len(gt_bboxes_) == 0:
                image_box_prob = torch.zeros(anchors.size(0), config.num_classes-1).type_as(bbox_preds_)
            else:
                # box_localization: a_{j}^{loc}, shape: [j, 4]
                pred_boxes = bbox_transform_inv_opr(anchors, bbox_preds_)

                # object_box_iou: IoU_{ij}^{loc}, shape: [i, j]
                object_box_iou = box_overlap_opr(gt_bboxes_, pred_boxes)

                # object_box_prob: P{a_{j} -> b_{i}}, shape: [i, j]
                t1 = config.bbox_thr
                t2 = object_box_iou.max(dim=1, keepdim=True).values.clamp(min=t1 + 1e-12)
                object_box_prob = ((object_box_iou - t1) / (t2 - t1)).clamp(min=0, max=1)

                # object_cls_box_prob: P{a_{j} -> b_{i}}, shape: [i, c, j]
                num_obj = gt_labels_.size(0)
                indices = torch.stack([torch.arange(num_obj).type_as(gt_labels_), gt_labels_],dim=0)
                object_cls_box_prob = torch.sparse.FloatTensor(indices, object_box_prob)
                                        
                # image_box_iou: P{a_{j} \in A_{+}}, shape: [c, j]
                box_cls_prob = torch.sparse.sum(object_cls_box_prob, dim=0).to_dense()
                indices = torch.nonzero(box_cls_prob, as_tuple=False).t_()
                if indices.numel() == 0:
                    image_box_prob = torch.zeros(anchors.size(0), config.num_classes-1).type_as(object_box_prob)
                else:
                    nonzero_box_prob = torch.where((gt_labels_.unsqueeze(dim=-1) == indices[0]),
                                        object_box_prob[:, indices[1]],
                                        torch.tensor([0]).type_as(object_box_prob)).max(dim=0).values
                    # upmap to shape [j, c]
                    image_box_prob = torch.sparse.FloatTensor(
                                        indices.flip([0]),
                                        nonzero_box_prob,
                                        size=(anchors.size(0), config.num_classes-1)).to_dense()
            box_prob.append(image_box_prob)
        # construct bags for objects
        match_quality_matrix = box_overlap_opr(gt_bboxes_, anchors)
        _, matched = torch.topk(match_quality_matrix, config.pre_anchor_topk, dim=1, sorted=False)
        del match_quality_matrix

        # matched_cls_prob: P_{ij}^{cls}
        matched_cls_prob = torch.gather(
                            cls_prob_[matched], 2,
                            gt_labels_.view(-1, 1, 1).repeat(1, config.pre_anchor_topk, 1)).squeeze(2)

        # matched_box_prob: P_{ij}^{loc}
        matched_anchors = anchors[matched]
        matched_gt_bboxes = gt_bboxes_.unsqueeze(dim=1).expand_as(matched_anchors)
        matched_object_targets = bbox_transform_opr(matched_anchors.reshape(-1, 4),
            matched_gt_bboxes.reshape(-1, 4))
        matched_object_targets = matched_object_targets.reshape(-1, config.pre_anchor_topk, 4)
        loss_reg = smooth_l1_loss(
                bbox_preds_[matched],
                matched_object_targets,
                config.smooth_l1_beta)
        matched_box_prob = torch.exp(-loss_reg)
        num_pos = num_pos + len(gt_bboxes_)
        positive_losses.append(positive_bag_loss(matched_cls_prob, matched_box_prob))
    positive_loss = torch.cat(positive_losses).sum() / max(1, num_pos)

    # box_prob: P{a_{j} \in A_{+}}
    box_prob = torch.stack(box_prob, dim=0)

    # negative_loss:
    # \sum_{j}{ FL((1 - P{a_{j} \in A_{+}}) * (1 - P_{j}^{bg})) } / n||B||
    negative_loss = negative_bag_loss(cls_prob, box_prob).sum() / max(1, num_pos * config.pre_anchor_topk)

    # avoid the absence of gradients in regression subnet
    # when no ground-truth in a batch
    if num_pos == 0:
        positive_loss = bbox_preds.sum() * 0

    losses = {
        'positive_bag_loss': positive_loss,
        'negative_bag_loss': negative_loss
    }
    return losses

def freeanchor_vpd_loss(anchors, cls_prob, bbox_preds, bbox_vpd_preds, gt_boxes, im_info):
    gt_labels, gt_bboxes = [], []
    cls_prob = cls_prob.reshape(config.train_batch_per_gpu, -1, config.num_classes-1)
    bbox_preds = bbox_preds.reshape(config.train_batch_per_gpu, -1, 4)
    bbox_vpd_preds = bbox_vpd_preds.reshape(config.train_batch_per_gpu, -1, 4)
    gt_boxes = [gt_boxes[bid, :int(im_info[bid, 5]), :] for bid in range(config.train_batch_per_gpu)]
    for gt_box in gt_boxes:
        obj_mask = torch.where(gt_box[:, -1] == 1)[0] 
        gt_labels.append(torch.zeros_like(gt_box[obj_mask, -1]).long())
        gt_bboxes.append(gt_box[obj_mask, :4])
    box_prob = []
    num_pos = 0
    positive_losses = []
    for _, (gt_labels_, gt_bboxes_, cls_prob_, bbox_preds_, bbox_vpd_preds_) in \
                        enumerate(zip(gt_labels, gt_bboxes, cls_prob, bbox_preds, bbox_vpd_preds)):
        with torch.no_grad():
            if len(gt_bboxes_) == 0:
                image_box_prob = torch.zeros(anchors.size(0), config.num_classes-1).type_as(bbox_preds_)
            else:
                # box_localization: a_{j}^{loc}, shape: [j, 4]
                pred_boxes = bbox_transform_inv_opr(anchors, bbox_preds_)

                # object_box_iou: IoU_{ij}^{loc}, shape: [i, j]
                object_box_iou = box_overlap_opr(gt_bboxes_, pred_boxes)

                # object_box_prob: P{a_{j} -> b_{i}}, shape: [i, j]
                t1 = config.bbox_thr
                t2 = object_box_iou.max(dim=1, keepdim=True).values.clamp(min=t1 + 1e-12)
                object_box_prob = ((object_box_iou - t1) / (t2 - t1)).clamp(min=0, max=1)

                # object_cls_box_prob: P{a_{j} -> b_{i}}, shape: [i, c, j]
                num_obj = gt_labels_.size(0)
                indices = torch.stack([torch.arange(num_obj).type_as(gt_labels_), gt_labels_],dim=0)
                object_cls_box_prob = torch.sparse.FloatTensor(indices, object_box_prob)
                                        
                # image_box_iou: P{a_{j} \in A_{+}}, shape: [c, j]
                box_cls_prob = torch.sparse.sum(object_cls_box_prob, dim=0).to_dense()
                indices = torch.nonzero(box_cls_prob, as_tuple=False).t_()
                if indices.numel() == 0:
                    image_box_prob = torch.zeros(anchors.size(0), config.num_classes-1).type_as(object_box_prob)
                else:
                    nonzero_box_prob = torch.where((gt_labels_.unsqueeze(dim=-1) == indices[0]),
                                        object_box_prob[:, indices[1]],
                                        torch.tensor([0]).type_as(object_box_prob)).max(dim=0).values
                    # upmap to shape [j, c]
                    image_box_prob = torch.sparse.FloatTensor(
                                        indices.flip([0]),
                                        nonzero_box_prob,
                                        size=(anchors.size(0), config.num_classes-1)).to_dense()
            box_prob.append(image_box_prob)
        # construct bags for objects
        match_quality_matrix = box_overlap_opr(gt_bboxes_, anchors)
        _, matched = torch.topk(match_quality_matrix, config.pre_anchor_topk, dim=1, sorted=False)
        del match_quality_matrix

        # matched_cls_prob: P_{ij}^{cls}
        matched_cls_prob = torch.gather(
                            cls_prob_[matched], 2,
                            gt_labels_.view(-1, 1, 1).repeat(1, config.pre_anchor_topk, 1)).squeeze(2)

        # matched_box_prob: P_{ij}^{loc}
        matched_anchors = anchors[matched]
        matched_gt_bboxes = gt_bboxes_.unsqueeze(dim=1).expand_as(matched_anchors)
        matched_object_targets = bbox_transform_opr(matched_anchors.reshape(-1, 4),
            matched_gt_bboxes.reshape(-1, 4))
        matched_object_targets = matched_object_targets.reshape(-1, config.pre_anchor_topk, 4)
        loss_reg = smooth_l1_loss(
                bbox_vpd_preds_[matched],
                matched_object_targets,
                config.smooth_l1_beta)
        matched_box_prob = torch.exp(-loss_reg)
        num_pos = num_pos + len(gt_bboxes_)
        positive_losses.append(positive_bag_loss(matched_cls_prob, matched_box_prob))
    positive_loss = torch.cat(positive_losses).sum() / max(1, num_pos)

    # box_prob: P{a_{j} \in A_{+}}
    box_prob = torch.stack(box_prob, dim=0)

    # negative_loss:
    # \sum_{j}{ FL((1 - P{a_{j} \in A_{+}}) * (1 - P_{j}^{bg})) } / n||B||
    negative_loss = negative_bag_loss(cls_prob, box_prob).sum() / max(1, num_pos * config.pre_anchor_topk)

    # avoid the absence of gradients in regression subnet
    # when no ground-truth in a batch
    if num_pos == 0:
        positive_loss = bbox_preds.sum() * 0

    losses = {
        'positive_bag_loss': positive_loss,
        'negative_bag_loss': negative_loss
    }
    return losses

def positive_bag_loss(matched_cls_prob, matched_box_prob):
    matched_prob = matched_cls_prob * matched_box_prob
    weight = 1 / torch.clamp(1 - matched_prob, 1e-12, None)
    weight = weight / weight.sum(dim=1).unsqueeze(dim=-1)
    bag_prob = (weight * matched_prob).sum(dim=1)
    return config.loss_box_alpha * F.binary_cross_entropy(bag_prob, \
        torch.ones_like(bag_prob), reduction='none')

def negative_bag_loss(cls_prob, box_prob):
    prob = cls_prob * (1 - box_prob)
    # There are some cases when neg_prob = 0.
    # This will cause the neg_prob.log() to be inf without clamp.
    prob = prob.clamp(min=1e-12, max=1 - 1e-12)
    negative_bag_loss = prob**config.loss_box_gamma * F.binary_cross_entropy(
        prob, torch.zeros_like(prob), reduction='none')
    return (1 - config.loss_box_alpha) * negative_bag_loss

def js_loss(dist, target, loss_weight):
    scale = (config.project.shape[1] - 1) / 2 / config.project[0,-1]
    acc = 1 / scale / 2
    target = (target.reshape(-1) + config.project[0,-1]) * scale
    target = target.clamp(min=EPS, max=2 * config.project[0,-1] * scale-EPS)
    idx_left = target.long()
    idx_right = idx_left + 1
    weight_left = idx_right.float() - target
    weight_right = target - idx_left.float()
    # target distribution
    target_dist = weight_left.new_full((weight_left.shape[0], \
        config.project.shape[1]), 0, dtype=torch.float32)
    target_dist[torch.arange(target_dist.shape[0]), idx_left] = weight_left
    target_dist[torch.arange(target_dist.shape[0]), idx_right] = weight_right
    # predict distribution
    mean= dist[:, :4].reshape(-1, 1)
    lstd= dist[:, 4:].reshape(-1, 1)
    Qg = torch.distributions.normal.Normal(mean, lstd.exp())
    project = torch.tensor(config.project).type_as(mean).repeat(mean.shape[0],1)
    pred_dist = Qg.cdf(project + acc) - Qg.cdf(project - acc)
    # JS distance
    total_dist = (target_dist + pred_dist) / 2
    loss1 = pred_dist * torch.log((pred_dist + EPS) / (total_dist + EPS))
    loss2 = target_dist * torch.log((target_dist + EPS) / (total_dist + EPS))
    loss = (loss1 + loss2).sum(dim=1) / 2
    return loss.reshape(-1, 4).sum(dim=1) * loss_weight