from megengine.data.transform.vision import functional as F
from megengine.data import transform as T
from megengine.data.dataset import Dataset
from typing import Sequence, Tuple
import os
import cv2
import numpy as np
import random
from PIL import Image, ImageDraw
import inspect
from data_augment import box_candidates, random_perspective
# import albumentations as A
try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None


class RandomHorizontalFlip(T.VisionTransform):
    r"""
    Horizontally flip the input data randomly with a given probability.

    :param p: probability of the input data being flipped. Default: 0.5
    :param order: the same with :class:`VisionTransform`.
    """

    def __init__(self, prob: float = 0.5, *, order=None):
        super().__init__(order)
        self.prob = prob

    def apply(self, input: Tuple):
        if 3 in input[2]:
            self._flipped = False
            # print("arr_l")
        else:
            # print("no arr_l")
            self._flipped = np.random.random() < self.prob
        self._w = self._get_image(input).shape[1]
        return super().apply(input)

    def _apply_image(self, image):
        if self._flipped:
            return F.flip(image, flipCode=1)
        return image

    def _apply_coords(self, coords):
        if self._flipped:
            coords[:, 0] = self._w - coords[:, 0]
        return coords

    def _apply_mask(self, mask):
        if self._flipped:
            return F.flip(mask, flipCode=1)
        return mask

class Albu(object):

    def __init__(self,
                 transforms,
                 bbox_params=None,
                 keymap=None,
                 update_pad_shape=False,
                 skip_img_without_anno=False):
        """
        Adds custom transformations from Albumentations lib.
        Please, visit `https://albumentations.readthedocs.io`
        to get more information.

        transforms (list): list of albu transformations
        bbox_params (dict): bbox_params for albumentation `Compose`
        keymap (dict): contains {'input key':'albumentation-style key'}
        skip_img_without_anno (bool): whether to skip the image
                                      if no ann left after aug
        """
        if Compose is None:
            raise RuntimeError('albumentations is not installed')

        self.transforms = transforms
        self.filter_lost_elements = False
        self.update_pad_shape = update_pad_shape
        self.skip_img_without_anno = skip_img_without_anno

        # A simple workaround to remove masks without boxes
        if (isinstance(bbox_params, dict) and 'label_fields' in bbox_params
                and 'filter_lost_elements' in bbox_params):
            self.filter_lost_elements = True
            self.origin_label_fields = bbox_params['label_fields']
            bbox_params['label_fields'] = ['idx_mapper']
            del bbox_params['filter_lost_elements']

        self.bbox_params = (
            self.albu_builder(bbox_params) if bbox_params else None)
        self.aug = Compose([self.albu_builder(t) for t in self.transforms],
                           bbox_params=self.bbox_params)

        # ['image', 'boxes', 'boxes_category', 'info']
        if not keymap:
            self.keymap_to_albu = {
                'image': 'image',
                'boxes': 'bboxes',
                'boxes_category': 'class_labels'
            }
        else:
            self.keymap_to_albu = keymap
        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}

    def albu_builder(self, cfg):
        """Import a module from albumentations.
        Inherits some of `build_from_cfg` logic.

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".
        Returns:
            obj: The constructed object.
        """
        assert isinstance(cfg, dict) and 'type' in cfg
        args = cfg.copy()

        obj_type = args.pop('type')
        if isinstance(obj_type, str):
            if albumentations is None:
                raise RuntimeError('albumentations is not installed')
            obj_cls = getattr(albumentations, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                'type must be a str or valid type, but got {}'.format(
                    type(obj_type)))

        if 'transforms' in args:
            args['transforms'] = [
                self.albu_builder(transform)
                for transform in args['transforms']
            ]

        return obj_cls(**args)

    @staticmethod
    def mapper(d, keymap):
        """
        Dictionary mapper.
        Renames keys according to keymap provided.

        Args:
            d (dict): old dict
            keymap (dict): {'old_key':'new_key'}
        Returns:
            dict: new dict.
        """
        updated_dict = {}
        for k, v in zip(d.keys(), d.values()):
            new_k = keymap.get(k, k)
            updated_dict[new_k] = d[k]
        return updated_dict

    def apply(self, input: Tuple):
        # dict to albumentations format

        keys = ['image', 'boxes', 'boxes_category', 'info']
        results = {keys[i]:inp for i,inp in enumerate(input)}
        img_shape = input[0].shape
        if len(results['boxes']) > 0:
            results['boxes'][:,0] = np.clip(results['boxes'][:,0], 0, img_shape[1] - 1)
            results['boxes'][:,1] = np.clip(results['boxes'][:,1], 0, img_shape[0] - 1)
            results['boxes'][:,2] = np.clip(results['boxes'][:,2], 0, img_shape[1] - 1)
            results['boxes'][:,3] = np.clip(results['boxes'][:,3], 0, img_shape[0] - 1)
            mask = [box[0]<box[2] and box[1]<box[3] for box in results['boxes']]
            results['boxes'] = results['boxes'][mask]
            results['boxes_category'] = results['boxes_category'][mask]

        if results['image'] is None or len(results['boxes']) == 0:
            return input
        # print(img_shape, results['boxes'])

        results = self.mapper(results, self.keymap_to_albu)

        if 'bboxes' in results:
            # to list of boxes
            if isinstance(results['bboxes'], np.ndarray):
                results['bboxes'] = [x for x in results['bboxes']]
            # add pseudo-field for filtration
            if self.filter_lost_elements:
                results['idx_mapper'] = np.arange(len(results['bboxes']))

        results = self.aug(**results)

        if 'bboxes' in results:
            if isinstance(results['bboxes'], list):
                results['bboxes'] = np.array(
                    results['bboxes'], dtype=np.float32)
            results['bboxes'] = results['bboxes'].reshape(-1, 4)

            # filter label_fields
            if self.filter_lost_elements:

                for label in self.origin_label_fields:
                    results[label] = np.array(
                        [results[label][i] for i in results['idx_mapper']])

                if (not len(results['idx_mapper']) and self.skip_img_without_anno):
                    return input

        # back to the original format
        results = self.mapper(results, self.keymap_back)

        # update final shape
        if self.update_pad_shape:
            results['pad_shape'] = results['img'].shape
        output = (
            results['image'],
            results['boxes'],
            results['boxes_category'],
            input[-1]
        )

        # self.vis_aug(results['image'], results['boxes'])
        # cv2.imwrite("./2.jpg",results['image'])
        # print(input[0].shape, output[0].shape)
        
        return output

    def vis_aug(self, img, bboxes):
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        for bbox in bboxes:
            draw.rectangle(bbox, outline='red')
        if not os.path.exists("./vis"):
            os.makedirs("./vis")
        idx = random.randint(1,100)
        # draw.show()
        img.save("./vis/" + str(idx) + ".jpg")

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(transforms={})'.format(self.transforms)
        return repr_str


class GridMask(object):
    def __init__(self, prob = 1., use_h=True, use_w=True, rotate = 1, offset=False, ratio = 0.5, mode=0):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode=mode
        self.st_prob = prob
        self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch

    def apply(self, input: Tuple):
        if np.random.rand() > self.prob:
            return input
        img= input[0]
        h,w,_ = img.shape
        # self.d1 = 2
        self.d1 = 16
        # self.d2 = min(h, w)
        self.d2 = min(512, h, w)
        hh = int(1.5*h)
        ww = int(1.5*w)
        d = np.random.randint(self.d1, self.d2)
        #d = self.d
        # self.l = int(d*self.ratio+0.5)
        if self.ratio == 1:
            self.l = np.random.randint(2, d)
        else:
            self.l = min(max(int(d*self.ratio+0.5),1),d-1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)

        if self.use_h:
            for i in range(hh//d):
                s = d*i + st_h
                t = min(s+self.l, hh)
                mask[s:t,:] *= 0
        if self.use_w:
            for i in range(ww//d):
                s = d*i + st_w
                t = min(s+self.l, ww)
                mask[:,s:t] *= 0
        mask = 1 - mask
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        # mask = 1*(np.random.randint(0,3,[hh,ww])>0)
        mask = mask[(hh-h)//2:(hh-h)//2+h, (ww-w)//2:(ww-w)//2+w]
        if self.mode == 1:
            mask = 1-mask
        mask = mask[:,:,np.newaxis]
        if self.offset:
            offset =2 * (np.random.rand(h,w) - 0.5)
            offset = (1 - mask) * offset
            img = img * mask + offset
        else:
            img = img * mask 

        output = [img]
        output.extend(input[1 : ])
        output = tuple(output)

        # self.vis_aug(img, input[1])

        return output
    
    def vis_aug(self, img, bboxes):
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        for bbox in bboxes:
            draw.rectangle(bbox, outline='red')
        if not os.path.exists("./vis_grid"):
            os.makedirs("./vis_grid")
        idx = random.randint(1,100)
        # draw.show()
        img.save("./vis_grid/" + str(idx) + ".jpg")

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(ratio={}, mode={}, prob={})'.format(
            self.ratio, self.mode, self.prob)
        return repr_str


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox

def get_mosaic_coordinate(mosaic_image, mosaic_index, xc, yc, w, h, input_h, input_w):
    # TODO update doc
    # index0 to top left part of image
    if mosaic_index == 0:
        x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
        small_coord = w - (x2 - x1), h - (y2 - y1), w, h
    # index1 to top right part of image
    elif mosaic_index == 1:
        x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
        small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
    # index2 to bottom left part of image
    elif mosaic_index == 2:
        x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
        small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
    # index2 to bottom right part of image
    elif mosaic_index == 3:
        x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)  # noqa
        small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
    return (x1, y1, x2, y2), small_coord

class MosaicDetection(Dataset):
    """Detection dataset wrapper that performs mixup for normal dataset."""

    def __init__(
        self, dataset, img_size, mosaic=True, preproc=None,
        degrees=10.0, translate=0.1, scale=(0.5, 1.5), mscale=(0.5, 1.5),
        shear=2.0, perspective=0.0, enable_mixup=True, *args
    ):
        """

        Args:
            dataset(Dataset) : Pytorch dataset object.
            img_size (tuple):
            mosaic (bool): enable mosaic augmentation or not.
            preproc (func):
            degrees (float):
            translate (float):
            scale (tuple):
            mscale (tuple):
            shear (float):
            perspective (float):
            enable_mixup (bool):
            *args(tuple) : Additional arguments for mixup random sampler.
        """
        super().__init__()
        self._dataset = dataset
        self.preproc = preproc
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.mixup_scale = mscale
        self.enable_mosaic = mosaic
        self.enable_mixup = enable_mixup
        self.input_dim = img_size

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):

        # if self.enable_mosaic and random.uniform(0, 1) < 0.75:
        if self.enable_mosaic:
            mosaic_labels = []
            input_dim = self.input_dim
            input_h, input_w = input_dim[0], input_dim[1]

            # yc, xc = s, s  # mosaic center x, y
            yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
            xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

            # 3 additional image indices
            indices = [idx] + [random.randint(0, len(self._dataset) - 1) for _ in range(3)]

            for i_mosaic, index in enumerate(indices):
                origin_img, origin_boxes, boxes_category, info = self._dataset.__getitem__(index)
                img = origin_img
                _labels = np.concatenate([origin_boxes, boxes_category[:,np.newaxis]],-1)

                h0, w0 = img.shape[:2]  # orig hw
                scale = min(1. * input_h / h0, 1. * input_w / w0)
                img = cv2.resize(
                    img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR
                )
                # generate output mosaic image
                (h, w, c) = img.shape[:3]
                if i_mosaic == 0:
                    mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)

                # suffix l means large image, while s means small image in mosaic aug.
                (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                    mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w
                )

                mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
                padw, padh = l_x1 - s_x1, l_y1 - s_y1

                labels = _labels.copy()
                # Normalized xywh to pixel xyxy format
                if _labels.size > 0:
                    labels[:, 0] = scale * _labels[:, 0] + padw
                    labels[:, 1] = scale * _labels[:, 1] + padh
                    labels[:, 2] = scale * _labels[:, 2] + padw
                    labels[:, 3] = scale * _labels[:, 3] + padh
                mosaic_labels.append(labels)

            if len(mosaic_labels):
                mosaic_labels = np.concatenate(mosaic_labels, 0)
                np.clip(mosaic_labels[:, 0], 0, 2 * input_w, out=mosaic_labels[:, 0])
                np.clip(mosaic_labels[:, 1], 0, 2 * input_h, out=mosaic_labels[:, 1])
                np.clip(mosaic_labels[:, 2], 0, 2 * input_w, out=mosaic_labels[:, 2])
                np.clip(mosaic_labels[:, 3], 0, 2 * input_h, out=mosaic_labels[:, 3])

            mosaic_img, mosaic_labels = random_perspective(
                mosaic_img,
                mosaic_labels,
                degrees=self.degrees,
                translate=self.translate,
                scale=self.scale,
                shear=self.shear,
                perspective=self.perspective,
                border=[-input_h // 2, -input_w // 2],
            )  # border to remove

            # -----------------------------------------------------------------
            # CopyPaste: https://arxiv.org/abs/2012.07177
            # -----------------------------------------------------------------
            if self.enable_mixup and not len(mosaic_labels) == 0 and random.uniform(0, 1) > 0.5:
                mosaic_img, mosaic_labels = self.mixup(mosaic_img, mosaic_labels, self.input_dim)
            if self.preproc is not None:
                mosaic_img, mosaic_labels = self.preproc(mosaic_img, mosaic_labels, self.input_dim)

            if not len(mosaic_labels) == 0:
                img = np.array(mosaic_img, dtype=origin_img.dtype)
                boxes = np.array(mosaic_labels[:,:4], dtype=origin_boxes.dtype)
                boxes_category = np.array(mosaic_labels[:,4], dtype=boxes_category.dtype)
                # self.vis_aug(img, boxes)
                return img, boxes, boxes_category, info
            else:
                return origin_img, origin_boxes, boxes_category, info

        else:
            self._dataset._input_dim = self.input_dim
            origin_img, origin_boxes, boxes_category, info = self._dataset.__getitem__(idx)
            mosaic_img = origin_img
            mosaic_labels = np.concatenate([origin_boxes, boxes_category[:,np.newaxis]],-1)
            if self.enable_mixup and not len(mosaic_labels) == 0 and random.uniform(0, 1) > 0.5:
                mosaic_img, mosaic_labels = self.mixup(mosaic_img, mosaic_labels, self.input_dim)

            if not len(mosaic_labels) == 0:
                img = np.array(mosaic_img, dtype=origin_img.dtype)
                boxes = np.array(mosaic_labels[:,:4], dtype=origin_boxes.dtype)
                boxes_category = np.array(mosaic_labels[:,4], dtype=boxes_category.dtype)
                # self.vis_aug(img, boxes)
                return img, boxes, boxes_category, info
            else:
                return origin_img, origin_boxes, boxes_category, info

    def vis_aug(self, img, bboxes):
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        for bbox in bboxes:
            draw.rectangle(bbox, outline='red')
        if not os.path.exists("./vis_mosaic"):
            os.makedirs("./vis_mosaic")
        idx = random.randint(1,100)
        # draw.show()
        img.save("./vis_mosaic/" + str(idx) + ".jpg")

    def mixup(self, origin_img, origin_labels, input_dim):
        jit_factor = random.uniform(*self.mixup_scale)
        # FLIP = random.uniform(0, 1) > 0.5
        FLIP = False

        cp_labels = []
        while len(cp_labels) == 0:
            cp_index = random.randint(0, self.__len__() - 1)
            img, boxes, boxes_category, _ = self._dataset.__getitem__(cp_index)
            cp_labels = np.concatenate([boxes, boxes_category[:,np.newaxis]],-1)

        if len(img.shape) == 3:
            cp_img = np.ones((input_dim[0], input_dim[1], 3)) * 114.0
        else:
            cp_img = np.ones(input_dim) * 114.0
        cp_scale_ratio = min(input_dim[0] / img.shape[0], input_dim[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        cp_img[
            : int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)
        ] = resized_img
        cp_img = cv2.resize(
            cp_img,
            (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
        )
        cp_scale_ratio *= jit_factor
        if FLIP:
            cp_img = cp_img[:, ::-1, :]

        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3)
        ).astype(np.uint8)
        padded_img[:origin_h, :origin_w] = cp_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[
            y_offset: y_offset + target_h, x_offset: x_offset + target_w
        ]

        cp_bboxes_origin_np = adjust_box_anns(
            cp_labels[:, :4].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h
        )
        if FLIP:
            cp_bboxes_origin_np[:, 0::2] = (
                origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1]
            )
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        cp_bboxes_transformed_np[:, 0::2] = np.clip(
            cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w
        )
        cp_bboxes_transformed_np[:, 1::2] = np.clip(
            cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h
        )
        keep_list = box_candidates(cp_bboxes_origin_np.T, cp_bboxes_transformed_np.T, 5)

        if keep_list.sum() >= 1.0:
            cls_labels = cp_labels[keep_list, 4:5].copy()
            box_labels = cp_bboxes_transformed_np[keep_list]
            labels = np.hstack((box_labels, cls_labels))
            origin_labels = np.vstack((origin_labels, labels))
            origin_img = origin_img.astype(np.float32)
            origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)

        return origin_img.astype(np.uint8), origin_labels

    def get_img_info(self, index):
        return self._dataset.get_img_info(index)

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

class ClassBalancedDataset(Dataset):
    """
    Args:
        dataset (:obj:`CustomDataset`): The dataset to be repeated.
        oversample_thr (float): frequency threshold below which data is
            repeated. For categories with ``f_c >= oversample_thr``, there is
            no oversampling. For categories with ``f_c < oversample_thr``, the
            degree of oversampling following the square-root inverse frequency
            heuristic above.
        filter_empty_gt (bool, optional): If set true, images without bounding
            boxes will not be oversampled. Otherwise, they will be categorized
            as the pure background class and involved into the oversampling.
            Default: True.
    """
    def __init__(self, dataset, num_classes, oversample_thr, filter_empty_gt=True):
        self.dataset = dataset
        self.oversample_thr = oversample_thr
        self.filter_empty_gt = filter_empty_gt
        self.CLASSES = num_classes

        repeat_factors = self._get_repeat_factors(dataset, oversample_thr)
        repeat_indices = []
        for dataset_idx, repeat_factor in enumerate(repeat_factors):
            repeat_indices.extend([dataset_idx] * math.ceil(repeat_factor))#向上取整
        self.repeat_indices = repeat_indices

        flags = []
        if hasattr(self.dataset, 'flag'):
            for flag, repeat_factor in zip(self.dataset.flag, repeat_factors):
                flags.extend([flag] * int(math.ceil(repeat_factor)))
            assert len(flags) == len(repeat_indices)
        self.flag = np.asarray(flags, dtype=np.uint8)

    def _get_repeat_factors(self, dataset, repeat_thr):
        """Get repeat factor for each images in the dataset.
        Args:
            dataset (:obj:`CustomDataset`): The dataset
            repeat_thr (float): The threshold of frequency. If an image
                contains the categories whose frequency below the threshold,
                it would be repeated.
        Returns:
            list[float]: The repeat factors for each images in the dataset.
        """

        # 1. For each category c, compute the fraction # of images
        #   that contain it: f(c)
        category_freq = defaultdict(int) # 当key不存在时，默认返回0
        num_images = len(dataset)
        for idx in range(num_images):
            cat_ids_rep = []
            img_id = self.dataset.ids[idx]
            for anns in self.dataset.img_to_anns[img_id]:
                cat_ids_rep.append(anns["category_id"])
            cat_ids = set(cat_ids_rep)
            if len(cat_ids) == 0 and not self.filter_empty_gt:
                cat_ids = set([len(self.CLASSES)])
            for cat_id in cat_ids:
                category_freq[cat_id] += 1
        for k, v in category_freq.items():
            category_freq[k] = v / num_images
        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t/f(c)))
        category_repeat = {
            cat_id: max(1.0, math.sqrt(repeat_thr / cat_freq))
            for cat_id, cat_freq in category_freq.items()
        }
        # 3. For each image I, compute the image-level repeat factor:
        #    r(I) = max_{c in I} r(c)
        repeat_factors = []
        for idx in range(num_images):
            cat_ids_rep = []
            img_id = self.dataset.ids[idx]
            for anns in self.dataset.img_to_anns[img_id]:
                cat_ids_rep.append(anns["category_id"])
            cat_ids = set(cat_ids_rep)
            if len(cat_ids) == 0 and not self.filter_empty_gt:
                cat_ids = set([len(self.CLASSES)]) 
            repeat_factor = 1
            if len(cat_ids) > 0:
                repeat_factor = max(
                    {category_repeat[cat_id]
                     for cat_id in cat_ids})
            repeat_factors.append(repeat_factor)
        return repeat_factors

    def __getitem__(self, idx):
        ori_index = self.repeat_indices[idx]
        return self.dataset[ori_index]

    def __len__(self):
        """Length after repetition."""
        return len(self.repeat_indices)
    
    def get_img_info(self, index):
        ori_index = self.repeat_indices[idx]
        return self.dataset.get_img_info(ori_index)
