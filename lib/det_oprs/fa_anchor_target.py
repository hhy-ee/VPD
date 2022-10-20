import torch

import numpy as np
from config import config
from det_oprs.bbox_opr import box_overlap_opr, bbox_transform_opr

INF = 100000000

@torch.no_grad()
def fa_anchor_target(anchors, gt_boxes, im_info, top_k):
    return_labels = []
    return_bbox_targets = []
    num_bboxes = anchors.size(0)
    # get per image proposals and gt_boxes
    for bid in range(config.train_batch_per_gpu):
        gt_boxes_perimg = gt_boxes[bid, :int(im_info[bid, 5]), :]
        obj_mask = torch.where(gt_boxes_perimg[:, -1] == 1)[0]
        gt_boxes_perimg = gt_boxes_perimg[obj_mask]
        num_gt = gt_boxes_perimg.size(0)
        overlaps = box_overlap_opr(anchors, gt_boxes_perimg[:, :-1])
        overlaps_inf = torch.full_like(overlaps, -INF).t().contiguous().view(-1)
        gt_assignment = overlaps.new_full((num_bboxes, ), 0, dtype=torch.long)
        # labels
        labels = gt_assignment.new_full((num_bboxes, ), 0, dtype=torch.float32)
        if num_gt != 0:
        # gt max and indices
            candidate_overlaps, candidate_idxs = overlaps.topk(top_k, dim=0, sorted=True)
            del overlaps
            for gt_idx in range(num_gt):
                candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
            candidate_idxs = candidate_idxs.view(-1)
            overlaps_inf[candidate_idxs] = candidate_overlaps.view(-1)
            overlaps_inf = overlaps_inf.view(num_gt, -1).t()
            max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
            gt_assignment[max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1
            pos_inds = torch.nonzero(gt_assignment > 0, as_tuple=False).squeeze()
            # cons labels
            if pos_inds.numel() > 0:
                labels[pos_inds] = gt_boxes_perimg[gt_assignment[pos_inds]-1, -1]
            # cons bbox targets
            target_boxes = gt_boxes_perimg[gt_assignment - 1, :4]
            bbox_targets = bbox_transform_opr(anchors, target_boxes)
            labels = labels.reshape(-1, 1)
            bbox_targets = bbox_targets.reshape(-1, 4)
        else:
            labels = labels.reshape(-1, 1)
            bbox_targets = torch.zeros_like(anchors)
        return_labels.append(labels)
        return_bbox_targets.append(bbox_targets)

    if config.train_batch_per_gpu == 1:
        return labels, bbox_targets
    else:
        return_labels = torch.cat(return_labels, axis=0)
        return_bbox_targets = torch.cat(return_bbox_targets, axis=0)
        return return_labels, return_bbox_targets

@torch.no_grad()
def fa_anchor_target_ign(anchors, gt_boxes, im_info, top_k):
    return_labels = []
    return_bbox_targets = []
    num_bboxes = anchors.size(0)
    # get per image proposals and gt_boxes
    for bid in range(config.train_batch_per_gpu):
        gt_boxes_perimg = gt_boxes[bid, :int(im_info[bid, 5]), :]
        obj_mask = torch.where(gt_boxes_perimg[:, -1] == 1)[0]
        ign_mask = torch.where(gt_boxes_perimg[:, -1] == -1)[0]
        ig_boxes_perimg = gt_boxes_perimg[ign_mask]
        gt_boxes_perimg = gt_boxes_perimg[obj_mask]
        num_gt = gt_boxes_perimg.size(0)
        overlaps = box_overlap_opr(anchors, gt_boxes_perimg[:, :-1])
        overlaps_inf = torch.full_like(overlaps, -INF).t().contiguous().view(-1)
        gt_assignment = overlaps.new_full((num_bboxes, ), 0, dtype=torch.long)
        # labels
        labels = gt_assignment.new_full((num_bboxes, ), 0, dtype=torch.float32)
        if ig_boxes_perimg.shape[0] != 0:
            ignore_overlaps = box_ioa_opr(anchors, ig_boxes_perimg[:, :-1])
            ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            ignore_idxs = ignore_max_overlaps > config.ignore_ioa_thr
            overlaps[ignore_idxs, :] = INF
            gt_assignment[ignore_idxs] = -1
        if num_gt != 0:
        # gt max and indices
            candidate_overlaps, candidate_idxs = overlaps.topk(top_k, dim=0, sorted=True)
            del overlaps
            for gt_idx in range(num_gt):
                candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
            candidate_idxs = candidate_idxs.view(-1)
            overlaps_inf[candidate_idxs] = candidate_overlaps.view(-1)
            overlaps_inf = overlaps_inf.view(num_gt, -1).t()
            max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
            gt_assignment[max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1
            pos_inds = torch.nonzero(gt_assignment > 0, as_tuple=False).squeeze()
            # cons labels
            if pos_inds.numel() > 0:
                labels[pos_inds] = gt_boxes_perimg[gt_assignment[pos_inds]-1, -1]
            # cons bbox targets
            target_boxes = gt_boxes_perimg[gt_assignment - 1, :4]
            bbox_targets = bbox_transform_opr(anchors, target_boxes)
            labels = labels.reshape(-1, 1)
            bbox_targets = bbox_targets.reshape(-1, 4)
        else:
            labels = labels.reshape(-1, 1)
            bbox_targets = torch.zeros_like(anchors)
        return_labels.append(labels)
        return_bbox_targets.append(bbox_targets)

    if config.train_batch_per_gpu == 1:
        return labels, bbox_targets
    else:
        return_labels = torch.cat(return_labels, axis=0)
        return_bbox_targets = torch.cat(return_bbox_targets, axis=0)
        return return_labels, return_bbox_targets