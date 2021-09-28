# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import models
albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.5,
        rotate_limit=30,
        interpolation=1,
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],
        p=0.1),
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),
    # dict(type='Cutout',num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, always_apply=False, p=0.5)
    ]

class CustomerConfig(models.ATSSConfig):
    def __init__(self):
        super().__init__()
        # ------------------------ dataset cfg ---------------------- #
        self.train_dataset = dict(
            name="traffic5",
            root="images",
            ann_file="annotations/train.json",
            remove_images_without_annotations=True,
        )
        self.test_dataset = dict(
            name="traffic5",
            root="images",
            ann_file="annotations/val.json",
            test_final_ann_file="annotations/test.json",
            remove_images_without_annotations=False,
        )
        self.num_classes = 5
        # self.backbone_freeze_at = 1

        # ------------------------ training cfg ---------------------- #
        self.basic_lr = 0.02 / 16
        self.max_epoch = 36
        self.lr_decay_stages = [24, 33]

        # self.backbone = "resnet101"
        self.backbone = "resnext101_32x8d"
        self.backbone_pretrained = True

        # self.backbone_dcn = True
        # self.backbone_stage_with_dcn = (False, False, True, True)
        self.backbone_dcn = False
        self.backbone_stage_with_dcn = (False, False, False, False)
        self.backbone_gcb = False
        self.backbone_stage_with_gcb = (False, False, False, False)

        self.train_image_short_size = (1500,)
        self.train_image_max_size = 2000
        self.test_image_short_size = 2100
        self.test_image_max_size = 2800

        self.test_max_boxes_per_image = 100
        self.test_cls_threshold = 0.05
        self.test_nms = 0.1

        self.TTA = True
        self.test_aug_short_size = 2400
        self.test_aug_max_size = (3100, 3200)
        self.test_size = [(2100,2700),(2100,2800),(2400,3200)]

        self.test_aug_flip = False

        self.TTA_crop = False
        self.test_crop_short_size = (900,)
        self.test_crop_max_size = 2000

        self.train_image_albu = dict(transforms=albu_train_transforms,
                                        bbox_params=dict(
                                            type='BboxParams',
                                            format='pascal_voc',
                                            label_fields=['class_labels'],
                                            min_visibility=0.0,
                                            filter_lost_elements=True),
                                        keymap= {
                                                'image': 'image',
                                                'boxes': 'bboxes',
                                                'boxes_category': 'class_labels'
                                                },
                                        update_pad_shape=False,
                                        skip_img_without_anno=True)

        self.train_mosaic = dict(img_size=(1500,2000), 
                                mosaic=True, 
                                preproc=None,
                                degrees=0.0, 
                                translate=0.1, 
                                scale=(0.5, 2.0),  #scale=(0.5, 1.5), 
                                mscale=(0.5, 2.0), #mscale=(0.5, 1.5),
                                shear=2.0, 
                                perspective=0.0, 
                                enable_mixup=True
                            )
        self.train_image_gridmask_prob = 0.5

        self.classbalanced = False
        self.oversample_thr = 0.25

        self.nr_images_epoch = 2226
        self.warm_iters = 100
        self.log_interval = 10


Net = models.ATSS
Cfg = CustomerConfig
