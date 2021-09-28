# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import os
import json
from collections import defaultdict

import cv2
import numpy as np

from megengine.data.dataset.vision.meta_vision import VisionDataset


def has_valid_annotation(anno, order):
    # if it"s empty, there is no annotation
    if len(anno) == 0:
        return False
    if "boxes" in order or "boxes_category" in order:
        if "bbox" not in anno[0]:
            return False
    return True


class Traffic5(VisionDataset):
    r"""
    Traffic Detection Challenge Dataset.
    """

    supported_order = (
        "image",
        "boxes",
        "boxes_category",
        "info",
    )

    def __init__(
        self, root, ann_file, remove_images_without_annotations=False, *, order=None
    ):
        super().__init__(root, order=order, supported_order=self.supported_order)

        with open(ann_file, "r") as f:
            dataset = json.load(f)

        self.imgs = dict()
        for img in dataset["images"]:
            self.imgs[img["id"]] = img

        self.img_to_anns = defaultdict(list)
        for ann in dataset["annotations"]:
            # for saving memory
            if (
                "boxes" not in self.order
                and "boxes_category" not in self.order
                and "bbox" in ann
            ):
                del ann["bbox"]
            if "polygons" not in self.order and "segmentation" in ann:
                del ann["segmentation"]
            self.img_to_anns[ann["image_id"]].append(ann)

        self.cats = dict()
        for cat in dataset["categories"]:
            self.cats[cat["id"]] = cat
        self.ids = list(sorted(self.imgs.keys()))

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                anno = self.img_to_anns[img_id]
                # filter crowd annotations
                anno = [obj for obj in anno if obj["iscrowd"] == 0]
                anno = [
                    obj for obj in anno if obj["bbox"][2] > 0 and obj["bbox"][3] > 0
                ]
                if has_valid_annotation(anno, order):
                    ids.append(img_id)
                    self.img_to_anns[img_id] = anno
                else:
                    del self.imgs[img_id]
                    del self.img_to_anns[img_id]
            self.ids = ids

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(sorted(self.cats.keys()))
        }

        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

    def __getitem__(self, index):
        img_id = self.ids[index]
        anno = self.img_to_anns[img_id]

        target = []
        for k in self.order:
            if k == "image":
                file_name = self.imgs[img_id]["file_name"]
                path = os.path.join(self.root, file_name)
                # print(path)
                image = cv2.imread(path, cv2.IMREAD_COLOR)
                target.append(image)
            elif k == "boxes":
                boxes = [obj["bbox"] for obj in anno]
                boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
                # transfer boxes from xywh to xyxy
                boxes[:, 2:] += boxes[:, :2]
                target.append(boxes)
            elif k == "boxes_category":
                boxes_category = [obj["category_id"] for obj in anno]
                boxes_category = [
                    self.json_category_id_to_contiguous_id[c] for c in boxes_category
                ]
                boxes_category = np.array(boxes_category, dtype=np.int32)
                target.append(boxes_category)
            elif k == "info":
                info = self.imgs[img_id]
                info = [info["height"], info["width"], info["file_name"], img_id]
                target.append(info)
            else:
                raise NotImplementedError

        return tuple(target)

    def __len__(self):
        return len(self.ids)

    def get_img_info(self, index):
        img_id = self.ids[index]
        img_info = self.imgs[img_id]
        return img_info

    class_names = (
        "red_tl",
        "arr_s",
        "arr_l",
        "no_driving_mark_allsort",
        "no_parking_mark",
    )

    classes_originID = {
        "red_tl": 0,
        "arr_s": 1,
        "arr_l": 2,
        "no_driving_mark_allsort": 3,
        "no_parking_mark": 4,
    }
