# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np

def bbox_area2(box):
    w = box[:, 2] - box[:, 0] + 1
    h = box[:, 3] - box[:, 1] + 1
    return w * h


def bbox_overlaps2(x, y):
    area_x = bbox_area2(x)
    area_y = bbox_area2(y)

    lt = np.maximum(x[:, None, :2], y[:, :2])  # [N,M,2]
    rb = np.minimum(x[:, None, 2:], y[:, 2:])  # [N,M,2]

    wh = (rb - lt) + 1  # [N,M,2]
    invalid = (wh[:, :, 0] <= 0) | (wh[:, :, 1] <= 0)

    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area_x[:, None] + area_y - inter)
    iou[invalid] = 0
    return iou

def topk_voting(nms_dets, dets, vote_thresh, k):
    top_dets = nms_dets.copy()
    top_boxes = nms_dets[:, :4]
    all_boxes = dets[:, :4]
    all_scores = dets[:, 4]
    top_to_all_overlaps = bbox_overlaps2(top_boxes, all_boxes)
    for i in range(nms_dets.shape[0]):
        inds_to_vote = np.where(top_to_all_overlaps[i] >= vote_thresh)[0]
        inds_to_vote = inds_to_vote[np.argsort(all_scores[inds_to_vote])[-k:]]
        boxes_to_vote = all_boxes[inds_to_vote, :]
        ws = all_scores[inds_to_vote]
        top_dets[i, 4] = np.average(ws)
        top_dets[i, :4] = np.average(boxes_to_vote, axis=0, weights=ws)

    return top_dets

def py_cpu_nms(dets, thresh):
    x1 = np.ascontiguousarray(dets[:, 0])
    y1 = np.ascontiguousarray(dets[:, 1])
    x2 = np.ascontiguousarray(dets[:, 2])
    y2 = np.ascontiguousarray(dets[:, 3])

    areas = (x2 - x1) * (y2 - y1)
    order = dets[:, 4].argsort()[::-1]
    keep = list()

    while order.size > 0:
        pick_idx = order[0]
        keep.append(pick_idx)
        order = order[1:]

        xx1 = np.maximum(x1[pick_idx], x1[order])
        yy1 = np.maximum(y1[pick_idx], y1[order])
        xx2 = np.minimum(x2[pick_idx], x2[order])
        yy2 = np.minimum(y2[pick_idx], y2[order])

        inter = np.maximum(xx2 - xx1, 0) * np.maximum(yy2 - yy1, 0)
        iou = inter / np.maximum(areas[pick_idx] + areas[order] - inter, 1e-5)

        order = order[iou <= thresh]

    return keep
