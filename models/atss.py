# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np

import megengine.functional as F
import megengine.module as M
import cv2
import layers
from layers.det import resnet
import copy


class ATSS(M.Module):
    """
    Implement ATSS (https://arxiv.org/abs/1912.02424).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.anchor_generator = layers.AnchorPointGenerator(
            cfg.num_anchors,
            strides=self.cfg.stride,
            offset=self.cfg.anchor_offset,
        )
        self.point_coder = layers.PointCoder()

        self.in_features = cfg.in_features

        # ----------------------- build backbone ------------------------ #
        bottom_up = getattr(resnet, cfg.backbone)(
            norm=layers.get_norm(cfg.backbone_norm), pretrained=cfg.backbone_pretrained
        )
        bottom_up = getattr(resnet, cfg.backbone)(
            norm=layers.get_norm(cfg.backbone_norm), pretrained=cfg.backbone_pretrained,
            dcn=cfg.backbone_dcn, stage_with_dcn=cfg.backbone_stage_with_dcn,
            gcb=cfg.backbone_gcb, stage_with_gcb=cfg.backbone_stage_with_gcb
        )
        del bottom_up.fc

        # ----------------------- build FPN ----------------------------- #
        self.backbone = layers.FPN(
            bottom_up=bottom_up,
            in_features=cfg.fpn_in_features,
            out_channels=cfg.fpn_out_channels,
            norm=cfg.fpn_norm,
            top_block=layers.LastLevelP6P7(
                cfg.fpn_top_in_channel, cfg.fpn_out_channels, cfg.fpn_top_in_feature
            ),
            strides=cfg.fpn_in_strides,
            channels=cfg.fpn_in_channels,
        )

        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]

        # ----------------------- build ATSS Head ----------------------- #
        self.head = layers.PointHead(cfg, feature_shapes)

    def preprocess_image(self, image):
        padded_image = layers.get_padded_tensor(image, 32, 0.0)
        normed_image = (
                               padded_image
                               - np.array(self.cfg.img_mean, dtype=np.float32)[None, :, None, None]
                       ) / np.array(self.cfg.img_std, dtype=np.float32)[None, :, None, None]
        return normed_image

    def get_hw_by_short_size(self, im_height, im_width, short_size, max_size):

        im_size_min = np.min([im_height, im_width])
        im_size_max = np.max([im_height, im_width])
        scale = (short_size + 0.0) / im_size_min
        if scale * im_size_max > max_size:
            scale = (max_size + 0.0) / im_size_max

        resized_height, resized_width = (
            int(round(im_height * scale)),
            int(round(im_width * scale)),
        )
        return resized_height, resized_width

    def preprocess_msimage(self, image, im_info, flip=False):

        original_height, original_width = im_info[:, 2], im_info[:, 3]
        ms_images = []
        ms_im_info = []
        # for short_size in self.cfg.test_aug_short_size:
        #     max_size = self.cfg.test_aug_max_size
        #         for max_size in self.cfg.test_aug_max_size:
        #             short_size = self.cfg.test_aug_short_size
        for (short_size, max_size) in self.cfg.test_size:
            batch_img = []
            batch_im_info = []
            for i in range(len(image)):
                img = np.ascontiguousarray(image[i].transpose(1, 2, 0), dtype=np.float32)  # tensor to image
                resized_height, resized_width = self.get_hw_by_short_size(
                    np.int(original_height[i]), np.int(original_width[i]), short_size, max_size
                )
                resized_img = cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR, )
                resized_img = cv2.flip(resized_img, 1) if flip else resized_img
                trans_img = np.ascontiguousarray(
                    resized_img.transpose(2, 0, 1)[None, :, :, :], dtype=np.float32
                )
                im_info = np.array(
                    [(resized_height, resized_width, original_height, original_width)],
                    dtype=np.float32,
                )
                batch_img.append(trans_img)
                batch_im_info.append(im_info)
            batch_img = np.concatenate(batch_img, 0)
            batch_im_info = np.concatenate(batch_im_info, 0)

            padded_image = layers.get_padded_tensor(batch_img, 32, 0.0)
            normed_image = (
                                   padded_image
                                   - np.array(self.cfg.img_mean, dtype=np.float32)[None, :, None, None]
                           ) / np.array(self.cfg.img_std, dtype=np.float32)[None, :, None, None]
            ms_images.append(normed_image)
            ms_im_info.append(batch_im_info)

        return ms_images, ms_im_info

    def preprocess_redlightimage(self, image, im_info):

        original_height, original_width = im_info[:, 2], im_info[:, 3]
        ms_images = []
        ms_im_info = []
        for short_size in self.cfg.test_aug_short_size:
            max_size = self.cfg.test_aug_max_size

            batch_img = []
            batch_im_info = []
            for i in range(len(image)):
                img = np.ascontiguousarray(image[i].transpose(1, 2, 0), dtype=np.float32)  # tensor to image
                height = int(im_info[i, 0] // 2)
                width = int(im_info[i, 1] // 2)
                left = int(im_info[i, 1] // 4)
                top = int(im_info[i, 0] // 6)
                cropped_img = img[top: top + height, left: left + width]
                resized_height, resized_width = self.get_hw_by_short_size(
                    np.int(height), np.int(width), short_size, max_size
                )
                resized_img = cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR, )
                trans_img = np.ascontiguousarray(
                    resized_img.transpose(2, 0, 1)[None, :, :, :], dtype=np.float32
                )
                im_info = np.array(
                    [(resized_height, resized_width, original_height // 2, original_width // 2)],
                    dtype=np.float32,
                )
                batch_img.append(trans_img)
                batch_im_info.append(im_info)
            batch_img = np.concatenate(batch_img, 0)
            batch_im_info = np.concatenate(batch_im_info, 0)

            padded_image = layers.get_padded_tensor(batch_img, 32, 0.0)
            normed_image = (
                                   padded_image
                                   - np.array(self.cfg.img_mean, dtype=np.float32)[None, :, None, None]
                           ) / np.array(self.cfg.img_std, dtype=np.float32)[None, :, None, None]
            ms_images.append(normed_image)
            ms_im_info.append(batch_im_info)

        return ms_images, ms_im_info

    def aug_test(self, image, im_info, flip=False):
        ms_images, ms_im_info = self.preprocess_msimage(image, im_info, flip)
        aug_bboxes = []
        aug_scores = []
        for i in range(len(ms_images)):
            image, im_info = ms_images[i], ms_im_info[i]

            features = self.backbone(image)
            features = [features[f] for f in self.in_features]
            box_logits, box_offsets, box_ctrness = self.head(features)
            box_logits_list = [
                _.transpose(0, 2, 3, 1).reshape(image.shape[0], -1, self.cfg.num_classes)
                for _ in box_logits
            ]
            box_offsets_list = [
                _.transpose(0, 2, 3, 1).reshape(image.shape[0], -1, 4) for _ in box_offsets
            ]
            box_ctrness_list = [
                _.transpose(0, 2, 3, 1).reshape(image.shape[0], -1, 1) for _ in box_ctrness
            ]
            anchors_list = self.anchor_generator(features)
            all_level_box_logits = F.concat(box_logits_list, axis=1)
            all_level_box_offsets = F.concat(box_offsets_list, axis=1)
            all_level_box_ctrness = F.concat(box_ctrness_list, axis=1)

            assert image.shape[0] == 1

            pred_score, clipped_boxes = self.inference(all_level_box_logits, all_level_box_offsets,
                                                       all_level_box_ctrness, anchors_list, im_info)

            aug_scores.append(pred_score)
            aug_bboxes.append(clipped_boxes)

        aug_scores = F.concat(aug_scores, 0)
        aug_bboxes = F.concat(aug_bboxes, 0)

        # print(aug_scores, aug_bboxes.shape)

        return aug_scores, aug_bboxes

    def crop_test(self, image, im_info):
        ms_images, ms_im_info = self.preprocess_redlightimage(image, im_info)
        crop_bboxes = []
        crop_scores = []
        for i in range(len(ms_images)):
            image, im_info = ms_images[i], ms_im_info[i]

            features = self.backbone(image)
            features = [features[f] for f in self.in_features]
            box_logits, box_offsets, box_ctrness = self.head(features)
            box_logits_list = [
                _.transpose(0, 2, 3, 1).reshape(image.shape[0], -1, self.cfg.num_classes)
                for _ in box_logits
            ]
            box_offsets_list = [
                _.transpose(0, 2, 3, 1).reshape(image.shape[0], -1, 4) for _ in box_offsets
            ]
            box_ctrness_list = [
                _.transpose(0, 2, 3, 1).reshape(image.shape[0], -1, 1) for _ in box_ctrness
            ]
            anchors_list = self.anchor_generator(features)
            all_level_box_logits = F.concat(box_logits_list, axis=1)
            all_level_box_offsets = F.concat(box_offsets_list, axis=1)
            all_level_box_ctrness = F.concat(box_ctrness_list, axis=1)

            assert image.shape[0] == 1

            pred_score, clipped_boxes = self.inference(all_level_box_logits, all_level_box_offsets,
                                                       all_level_box_ctrness, anchors_list, im_info)

            clipped_boxes[:, 0] = clipped_boxes[:, 0] + np.array(im_info[0][3] // 2, dtype=np.float32)
            clipped_boxes[:, 2] = clipped_boxes[:, 2] + np.array(im_info[0][3] // 2, dtype=np.float32)
            clipped_boxes[:, 1] = clipped_boxes[:, 1] + np.array(im_info[0][2] // 3, dtype=np.float32)
            clipped_boxes[:, 3] = clipped_boxes[:, 3] + np.array(im_info[0][2] // 3, dtype=np.float32)

            crop_scores.append(pred_score)
            crop_bboxes.append(clipped_boxes)

        crop_scores = F.concat(crop_scores, 0)
        crop_bboxes = F.concat(crop_bboxes, 0)

        # print(acrop_scores, crop_bboxes.shape)

        return crop_scores, crop_bboxes

    def forward(self, image, im_info, gt_boxes=None):

        # ms_images, ms_im_info = self.preprocess_msimage(image, im_info)
        # image, im_info = ms_images[-1], ms_im_info[-1]
        if not self.training and self.cfg.TTA:
            original_height, original_width = im_info[0, 2], im_info[0, 3]
            aug_scores, aug_bboxes = self.aug_test(image, im_info, flip=False)
            if self.cfg.test_aug_flip:
                aug_scores_flip, aug_bboxes_flip = self.aug_test(image, im_info, flip=True)

                fliped_aug_scores = F.copy(aug_scores_flip)
                fliped_aug_scores[:, 2] = 0  ###left turn
                fliped_aug_bboxes = F.copy(aug_bboxes_flip)
                fliped_aug_bboxes[:, 0::4] = original_width - aug_bboxes_flip[:, 2::4]
                fliped_aug_bboxes[:, 2::4] = original_width - aug_bboxes_flip[:, 0::4]

                # aug_scores[:,:2] = (aug_scores[:,:2] + aug_scores_flip[:,:2]) /2
                # aug_scores[:,3:] = (aug_scores[:,3:] + aug_scores_flip[:,3:]) /2
                # aug_bboxes = (aug_bboxes + aug_bboxes_flip) /2
                # print(aug_bboxes[0], fliped_aug_bboxes[0], aug_bboxes_flip[0])

                aug_scores = F.concat([aug_scores, fliped_aug_scores], 0)
                aug_bboxes = F.concat([aug_bboxes, fliped_aug_bboxes], 0)

            return aug_scores, aug_bboxes

            # return aug_scores_flip, fliped_aug_bboxes

        if not self.training and self.cfg.TTA_crop:
            crop_scores, crop_bboxes = self.crop_test(image, im_info)

            return crop_scores, crop_bboxes

        image = self.preprocess_image(image)

        features = self.backbone(image)
        features = [features[f] for f in self.in_features]

        box_logits, box_offsets, box_ctrness = self.head(features)

        box_logits_list = [
            _.transpose(0, 2, 3, 1).reshape(image.shape[0], -1, self.cfg.num_classes)
            for _ in box_logits
        ]
        box_offsets_list = [
            _.transpose(0, 2, 3, 1).reshape(image.shape[0], -1, 4) for _ in box_offsets
        ]
        box_ctrness_list = [
            _.transpose(0, 2, 3, 1).reshape(image.shape[0], -1, 1) for _ in box_ctrness
        ]

        anchors_list = self.anchor_generator(features)

        all_level_box_logits = F.concat(box_logits_list, axis=1)
        all_level_box_offsets = F.concat(box_offsets_list, axis=1)
        all_level_box_ctrness = F.concat(box_ctrness_list, axis=1)

        if self.training:
            gt_labels, gt_offsets, gt_ctrness = self.get_ground_truth(
                anchors_list, gt_boxes, im_info[:, 4].astype(np.int32),
            )

            all_level_box_logits = all_level_box_logits.reshape(-1, self.cfg.num_classes)
            all_level_box_offsets = all_level_box_offsets.reshape(-1, 4)
            all_level_box_ctrness = all_level_box_ctrness.flatten()

            gt_labels = gt_labels.flatten()
            gt_offsets = gt_offsets.reshape(-1, 4)
            gt_ctrness = gt_ctrness.flatten()

            valid_mask = gt_labels >= 0
            fg_mask = gt_labels > 0
            num_fg = fg_mask.sum()

            gt_targets = F.zeros_like(all_level_box_logits)
            gt_targets[fg_mask, gt_labels[fg_mask] - 1] = 1

            loss_cls = layers.sigmoid_focal_loss(
                all_level_box_logits[valid_mask],
                gt_targets[valid_mask],
                alpha=self.cfg.focal_loss_alpha,
                gamma=self.cfg.focal_loss_gamma,
            ).sum() / F.maximum(num_fg, 1)

            if num_fg == 0:
                loss_bbox = all_level_box_offsets.sum() * 0.
                loss_ctr = all_level_box_ctrness.sum() * 0.
                total = loss_cls + loss_bbox + loss_ctr
                loss_dict = {
                    "total_loss": total,
                    "loss_cls": loss_cls,
                    "loss_bbox": loss_bbox,
                    "loss_ctr": loss_ctr
                }
                self.cfg.losses_keys = list(loss_dict.keys())
                return loss_dict

            sum_ctr = gt_ctrness[fg_mask].sum()
            # add detach() to avoid syncing across ranks in backward
            num_fg = layers.all_reduce_mean(num_fg).detach()
            sum_ctr = layers.all_reduce_mean(sum_ctr).detach()

            loss_bbox = (
                                layers.iou_loss(
                                    all_level_box_offsets[fg_mask],
                                    gt_offsets[fg_mask],
                                    box_mode="ltrb",
                                    loss_type=self.cfg.iou_loss_type,
                                ) * gt_ctrness[fg_mask]
                        ).sum() / F.maximum(sum_ctr, 1e-5) * self.cfg.loss_bbox_weight

            loss_ctr = layers.binary_cross_entropy(
                all_level_box_ctrness[fg_mask],
                gt_ctrness[fg_mask],
            ).sum() / F.maximum(num_fg, 1)

            total = loss_cls + loss_bbox + loss_ctr
            loss_dict = {
                "total_loss": total,
                "loss_cls": loss_cls,
                "loss_bbox": loss_bbox,
                "loss_ctr": loss_ctr,
            }
            self.cfg.losses_keys = list(loss_dict.keys())
            return loss_dict
        else:
            # currently not support multi-batch testing
            assert image.shape[0] == 1

            pred_score, clipped_boxes = self.inference(all_level_box_logits, all_level_box_offsets,
                                                       all_level_box_ctrness, anchors_list, im_info)
            # print(pred_score.shape, clipped_boxes.shape)

            return pred_score, clipped_boxes

    def inference(self, all_level_box_logits, all_level_box_offsets, all_level_box_ctrness, anchors_list, im_info):
        all_level_anchors = F.concat(anchors_list, axis=0)
        pred_boxes = self.point_coder.decode(
            all_level_anchors, all_level_box_offsets[0]
        )
        pred_boxes = pred_boxes.reshape(-1, 4)

        scale_w = im_info[0, 1] / im_info[0, 3]
        scale_h = im_info[0, 0] / im_info[0, 2]
        pred_boxes = pred_boxes / F.concat(
            [scale_w, scale_h, scale_w, scale_h], axis=0
        )
        clipped_boxes = layers.get_clipped_boxes(
            pred_boxes, im_info[0, 2:4]
        ).reshape(-1, 4)
        pred_score = F.sqrt(
            F.sigmoid(all_level_box_logits) * F.sigmoid(all_level_box_ctrness)
        )[0]

        return pred_score, clipped_boxes

    def get_ground_truth(self, anchors_list, batched_gt_boxes, batched_num_gts):
        labels_list = []
        offsets_list = []
        ctrness_list = []

        all_level_anchors = F.concat(anchors_list, axis=0)
        for bid in range(batched_gt_boxes.shape[0]):
            gt_boxes = batched_gt_boxes[bid, :batched_num_gts[bid]]

            ious = []
            candidate_idxs = []
            base = 0
            for stride, anchors_i in zip(self.cfg.stride, anchors_list):
                ious.append(layers.get_iou(
                    gt_boxes[:, :4],
                    F.concat([
                        anchors_i - stride * self.cfg.anchor_scale / 2,
                        anchors_i + stride * self.cfg.anchor_scale / 2,
                    ], axis=1)
                ))
                gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:4]) / 2
                distances = F.sqrt(
                    F.sum((F.expand_dims(gt_centers, axis=1) - anchors_i) ** 2, axis=2)
                )
                _, topk_idxs = F.topk(distances, self.cfg.anchor_topk)
                candidate_idxs.append(base + topk_idxs)
                base += anchors_i.shape[0]
            ious = F.concat(ious, axis=1)
            candidate_idxs = F.concat(candidate_idxs, axis=1)

            candidate_ious = F.gather(ious, 1, candidate_idxs)
            ious_thr = (F.mean(candidate_ious, axis=1, keepdims=True)
                        + F.std(candidate_ious, axis=1, keepdims=True))
            is_foreground = F.scatter(
                F.zeros(ious.shape), 1, candidate_idxs, F.ones(candidate_idxs.shape)
            ).astype(bool) & (ious >= ious_thr)

            is_in_boxes = F.min(self.point_coder.encode(
                all_level_anchors, F.expand_dims(gt_boxes[:, :4], axis=1)
            ), axis=2) > 0

            ious[~is_foreground] = -1
            ious[~is_in_boxes] = -1

            match_indices = F.argmax(ious, axis=0)
            gt_boxes_matched = gt_boxes[match_indices]
            anchor_max_iou = F.indexing_one_hot(ious, match_indices, axis=0)

            labels = gt_boxes_matched[:, 4].astype(np.int32)
            labels[anchor_max_iou == -1] = 0
            offsets = self.point_coder.encode(all_level_anchors, gt_boxes_matched[:, :4])

            left_right = offsets[:, [0, 2]]
            top_bottom = offsets[:, [1, 3]]
            ctrness = F.sqrt(
                F.clip(F.min(left_right, axis=1) / F.max(left_right, axis=1), lower=0)
                * F.clip(F.min(top_bottom, axis=1) / F.max(top_bottom, axis=1), lower=0)
            )

            labels_list.append(labels)
            offsets_list.append(offsets)
            ctrness_list.append(ctrness)

        return (
            F.stack(labels_list, axis=0).detach(),
            F.stack(offsets_list, axis=0).detach(),
            F.stack(ctrness_list, axis=0).detach(),
        )


class ATSSConfig:
    # pylint: disable=too-many-statements
    def __init__(self):
        self.backbone = "resnet50"
        self.backbone_pretrained = True
        self.backbone_norm = "FrozenBN"
        self.backbone_dcn = False
        self.backbone_stage_with_dcn = (False, False, False, False)
        self.backbone_gcb = False
        self.backbone_stage_with_gcb = (False, False, False, False)

        self.backbone_freeze_at = 2
        self.fpn_norm = None
        self.fpn_in_features = ["res3", "res4", "res5"]
        self.fpn_in_strides = [8, 16, 32]
        self.fpn_in_channels = [512, 1024, 2048]
        self.fpn_out_channels = 256
        self.fpn_top_in_feature = "p5"
        self.fpn_top_in_channel = 256

        # ------------------------ data cfg -------------------------- #
        self.train_dataset = dict(
            name="coco",
            root="train2017",
            ann_file="annotations/instances_train2017.json",
            remove_images_without_annotations=True,
        )
        self.test_dataset = dict(
            name="coco",
            root="val2017",
            ann_file="annotations/instances_val2017.json",
            remove_images_without_annotations=False,
        )
        self.num_classes = 80
        self.img_mean = [103.530, 116.280, 123.675]  # BGR
        self.img_std = [57.375, 57.120, 58.395]

        # ----------------------- net cfg ------------------------- #
        self.stride = [8, 16, 32, 64, 128]
        self.in_features = ["p3", "p4", "p5", "p6", "p7"]

        self.num_anchors = 1
        self.anchor_offset = 0.5

        self.anchor_scale = 8
        self.anchor_topk = 9
        self.class_aware_box = False
        self.cls_prior_prob = 0.01

        # ------------------------ loss cfg -------------------------- #
        self.focal_loss_alpha = 0.25
        self.focal_loss_gamma = 2
        self.iou_loss_type = "giou"
        self.loss_bbox_weight = 2.0
        self.num_losses = 4

        # ------------------------ training cfg ---------------------- #
        self.train_image_short_size = (640, 672, 704, 736, 768, 800)
        # self.train_image_short_size = (768,)
        self.train_image_max_size = 1333

        self.basic_lr = 0.01 / 16  # The basic learning rate for single-image0
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.log_interval = 20
        self.nr_images_epoch = 80000
        self.max_epoch = 54
        self.warm_iters = 500
        self.lr_decay_rate = 0.1
        self.lr_decay_stages = [42, 50]

        # ------------------------ testing cfg ----------------------- #
        self.test_image_short_size = 800
        self.test_image_max_size = 1333
        self.test_max_boxes_per_image = 100
        self.test_vis_threshold = 0.3
        self.test_cls_threshold = 0.05
        self.test_nms = 0.6
