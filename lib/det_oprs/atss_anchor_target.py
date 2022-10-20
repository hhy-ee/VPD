import torch

import numpy as np
from config import config
from det_oprs.bbox_opr import box_overlap_opr, bbox_transform_opr, bbox_transform_inv_opr, box_ioa_opr

INF = 10000

@torch.no_grad()
def atss_anchor_target(anchors, gt_boxes, num_levels, im_info):
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
        gt_assignment = overlaps.new_full((num_bboxes, ), 0, dtype=torch.long)
        overlaps_inf = torch.full_like(overlaps, -INF).t().contiguous().view(-1)

        # compute center distance between all bbox and gt
        gt_cx = (gt_boxes_perimg[:, 0] + gt_boxes_perimg[:, 2]) / 2.0
        gt_cy = (gt_boxes_perimg[:, 1] + gt_boxes_perimg[:, 3]) / 2.0
        gt_points = torch.stack((gt_cx, gt_cy), dim=1)
        bboxes_cx = (anchors[:, 0] + anchors[:, 2]) / 2.0
        bboxes_cy = (anchors[:, 1] + anchors[:, 3]) / 2.0
        bboxes_points = torch.stack((bboxes_cx, bboxes_cy), dim=1)
        distances = (bboxes_points[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt()
        
        # obtain ignore labels
        if ig_boxes_perimg.shape[0] != 0:
            ignore_overlaps = box_ioa_opr(anchors, ig_boxes_perimg[:, :-1])
            ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            ignore_idxs = ignore_max_overlaps > config.ignore_ioa_thr
            distances[ignore_idxs, :] = INF
            gt_assignment[ignore_idxs] = -1

        # Selecting candidates based on the center distance
        if num_gt != 0:
            candidate_idxs = []
            start_idx = 0
            for level, bboxes_per_level in enumerate(num_levels):
                # on each pyramid level, for each gt,
                # select k bbox whose center are closest to the gt center
                end_idx = start_idx + bboxes_per_level
                distances_per_level = distances[start_idx:end_idx, :]
                selectable_k = min(config.assign_topk, bboxes_per_level)
                _, topk_idxs_per_level = distances_per_level.topk(
                    selectable_k, dim=0, largest=False)
                candidate_idxs.append(topk_idxs_per_level + start_idx)
                start_idx = end_idx
            candidate_idxs = torch.cat(candidate_idxs, dim=0)

            # get corresponding iou for the these candidates, and compute the
            # mean and std, set mean + std as the iou threshold
            candidate_overlaps = overlaps[candidate_idxs, torch.arange(num_gt)]
            overlaps_mean_per_gt = candidate_overlaps.mean(0)
            overlaps_std_per_gt = candidate_overlaps.std(0)
            overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt
            is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]

            # limit the positive sample's center in gt
            for gt_idx in range(num_gt):
                candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
            ep_bboxes_cx = bboxes_cx.view(1, -1).expand(
                num_gt, num_bboxes).contiguous().view(-1)
            ep_bboxes_cy = bboxes_cy.view(1, -1).expand(
                num_gt, num_bboxes).contiguous().view(-1)
            candidate_idxs = candidate_idxs.view(-1)

            # calculate the left, top, right, bottom distance between positive
            # bbox center and gt side
            l_ = ep_bboxes_cx[candidate_idxs].view(-1, num_gt) - gt_boxes_perimg[:, 0]
            t_ = ep_bboxes_cy[candidate_idxs].view(-1, num_gt) - gt_boxes_perimg[:, 1]
            r_ = gt_boxes_perimg[:, 2] - ep_bboxes_cx[candidate_idxs].view(-1, num_gt)
            b_ = gt_boxes_perimg[:, 3] - ep_bboxes_cy[candidate_idxs].view(-1, num_gt)
            is_in_gts = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01
            is_pos = is_pos & is_in_gts

            # if an anchor box is assigned to multiple gts,
            # the one with the highest IoU will be selected.
            index = candidate_idxs.view(-1)[is_pos.view(-1)]
            overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
            overlaps_inf = overlaps_inf.view(num_gt, -1).t()
            max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
            gt_assignment[max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1
            # cons bbox targets
            target_boxes = gt_boxes_perimg[gt_assignment - 1, :4]
            bbox_targets = bbox_transform_opr(anchors, target_boxes)
            bbox_targets = bbox_targets.reshape(-1, 4)
        else:
            bbox_targets = torch.zeros_like(anchors)
        # cons anchor labels
        pos_inds = torch.nonzero(gt_assignment > 0, as_tuple=False).squeeze()
        ign_inds = torch.nonzero(gt_assignment < 0, as_tuple=False).squeeze()
        labels = gt_assignment.new_full((num_bboxes, ), 0, dtype=torch.float32)
        if pos_inds.numel() > 0:
            labels[pos_inds] = 1
        if ign_inds.numel() > 0:
            labels[ign_inds] = -1
        labels = labels.reshape(-1, 1)
        
        return_labels.append(labels)
        return_bbox_targets.append(bbox_targets)

    if config.train_batch_per_gpu == 1:
        return labels, bbox_targets
    else:
        return_labels = torch.cat(return_labels, axis=0)
        return_bbox_targets = torch.cat(return_bbox_targets, axis=0)
        return return_labels, return_bbox_targets

def centerness_target(anchors, bbox_targets):
    num_gt = bbox_targets.shape[0]
    if num_gt != 0:
        # only calculate pos centerness targets, otherwise there may be nan
        gts = bbox_transform_inv_opr(anchors, bbox_targets)
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        l_ = anchors_cx - gts[:, 0]
        t_ = anchors_cy - gts[:, 1]
        r_ = gts[:, 2] - anchors_cx
        b_ = gts[:, 3] - anchors_cy

        left_right = torch.stack([l_, r_], dim=1)
        top_bottom = torch.stack([t_, b_], dim=1)
        centerness = torch.sqrt(
            (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) *
            (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]))
        assert not torch.isnan(centerness).any()
    else:
        centerness = torch.zeros(0).type_as(bbox_targets)
    return centerness