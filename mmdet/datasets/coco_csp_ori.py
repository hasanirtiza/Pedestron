import numpy as np
from pycocotools.coco import COCO

from .custom import CustomDataset
from .registry import DATASETS
import cv2

import os.path as osp
import mmcv
from mmcv.parallel import DataContainer as DC
from .registry import DATASETS
from .transforms import (ImageTransform, BboxTransform, MaskTransform,
                         SegMapTransform, Numpy2Tensor)
from .utils import to_tensor, random_scale
from .extra_aug import ExtraAugmentation

INF = 1e8

@DATASETS.register_module
class CocoCSPORIDataset(CustomDataset):

    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
               'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
               'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
               'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')
    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 multiscale_mode='value',
                 size_divisor=None,
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0,
                 with_mask=True,
                 with_crowd=True,
                 with_label=True,
                 with_semantic_seg=False,
                 seg_prefix=None,
                 seg_scale_factor=1,
                 extra_aug=None,
                 resize_keep_ratio=True,
                 test_mode=False,
                 remove_small_box=False,
                 small_box_size=8,
                 strides=None,
                 regress_ranges=None,
                 upper_factor=None,
                 upper_more_factor=None,
                 with_width=False):
        # prefix of images path
        self.img_prefix = img_prefix

        # load annotations (and proposals)
        self.img_infos = self.load_annotations(ann_file)
        if proposal_file is not None:
            self.proposals = self.load_proposals(proposal_file)
        else:
            self.proposals = None
        # filter images with no annotation during training
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.img_infos = [self.img_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]

        # (long_edge, short_edge) or [(long1, short1), (long2, short2), ...]
        self.img_scales = img_scale if isinstance(img_scale,
                                                  list) else [img_scale]
        assert mmcv.is_list_of(self.img_scales, tuple)
        # normalization configs
        self.img_norm_cfg = img_norm_cfg

        # multi-scale mode (only applicable for multi-scale training)
        self.multiscale_mode = multiscale_mode
        assert multiscale_mode in ['value', 'range']

        # max proposals per image
        self.num_max_proposals = num_max_proposals
        # flip ratio
        self.flip_ratio = flip_ratio
        assert flip_ratio >= 0 and flip_ratio <= 1
        # padding border to ensure the image size can be divided by
        # size_divisor (used for FPN)
        self.size_divisor = size_divisor

        # with mask or not (reserved field, takes no effect)
        self.with_mask = with_mask
        # some datasets provide bbox annotations as ignore/crowd/difficult,
        # if `with_crowd` is True, then these info is returned.
        self.with_crowd = with_crowd
        # with label is False for RPN
        self.with_label = with_label
        # with semantic segmentation (stuff) annotation or not
        self.with_seg = with_semantic_seg
        # prefix of semantic segmentation map path
        self.seg_prefix = seg_prefix
        # rescale factor for segmentation maps
        self.seg_scale_factor = seg_scale_factor
        # in test mode or not
        self.test_mode = test_mode
        # remove small size box in gt
        self.remove_small_box = remove_small_box
        # the smallest box edge
        self.small_box_size = small_box_size
        # strides of FPN style outputs, used for generating groundtruth feature maps
        self.strides = strides
        # regress range of FPN style outputs, used for generating groundtruth feature maps
        self.regress_ranges = regress_ranges
        # upper factor for Irtiza's model which use three branches to predict upper box, full box, lower box
        self.upper_factor = upper_factor
        # split the upper box into more boxes
        self.upper_more_factor = upper_more_factor

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        # transforms
        self.img_transform = ImageTransform(
            size_divisor=self.size_divisor, **self.img_norm_cfg)
        self.bbox_transform = BboxTransform()
        self.mask_transform = MaskTransform()
        self.seg_transform = SegMapTransform(self.size_divisor)
        self.numpy2tensor = Numpy2Tensor()

        # if use extra augmentation
        if extra_aug is not None:
            self.extra_aug = ExtraAugmentation(**extra_aug)
        else:
            self.extra_aug = None

        # image rescale if keep ratio
        self.resize_keep_ratio = resize_keep_ratio

        #predict width
        self.with_width = with_width

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(ann_info, self.with_mask)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, ann_info, with_mask=True):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, mask_polys, poly_lens.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        # Two formats are provided.
        # 1. mask: a binary map of the same size of the image.
        # 2. polys: each mask consists of one or several polys, each poly is a
        # list of float.
        if with_mask:
            gt_masks = []
            gt_mask_polys = []
            gt_poly_lens = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            #if ann['area'] <= 0 or w < 1 or h < 1:
            #    continue
            if w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
            if with_mask:
                #create fake segmentation
                if "segmentation" not in ann:
                    bbox = ann['bbox']
                    ann['segmentation'] = [[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1],
                                           bbox[0], bbox[1] + bbox[3], bbox[0] + bbox[2], bbox[1] + bbox[3]]]
                gt_masks.append(self.coco.annToMask(ann))
                # cv2.imshow('', gt_masks[-1]*255)
                # cv2.waitKey(0)
                # print(gt_masks[-1].shape)
                mask_polys = [
                    p for p in ann['segmentation'] if len(p) >= 6
                ]  # valid polygons have >= 3 points (6 coordinates)
                poly_lens = [len(p) for p in mask_polys]
                gt_mask_polys.append(mask_polys)
                gt_poly_lens.extend(poly_lens)
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)

        if with_mask:
            ann['masks'] = gt_masks
            # poly format is not used in the current implementation
            ann['mask_polys'] = gt_mask_polys
            ann['poly_lens'] = gt_poly_lens
        return ann

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        # load image
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        # load proposals if necessary
        if self.proposals is not None:
            proposals = self.proposals[idx][:self.num_max_proposals]
            # TODO: Handle empty proposals properly. Currently images with
            # no proposals are just ignored, but they can be used for
            # training in concept.
            if len(proposals) == 0:
                return None
            if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposals.shape))
            if proposals.shape[1] == 5:
                scores = proposals[:, 4, None]
                proposals = proposals[:, :4]
            else:
                scores = None

        ann = self.get_ann_info(idx)
        gt_bboxes = ann['bboxes']
        gt_labels = ann['labels']
        if self.with_crowd:
            gt_bboxes_ignore = ann['bboxes_ignore']

        assert len(self.img_scales[0]) == 2 and isinstance(self.img_scales[0][0], int)

        img, gt_bboxes, gt_labels, gt_bboxes_ignore = augment(img, gt_bboxes, gt_labels, gt_bboxes_ignore, self.img_scales[0])
        ori_shape = img.shape[:2]
        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, img.shape[:2], False, keep_ratio=self.resize_keep_ratio)
        assert (scale_factor == 1)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=ori_shape,
            pad_shape=(0,0),
            scale_factor=1,
            flip=False,
            name=img_info['filename'],
        )

        pos_maps = []
        scale_maps = []
        offset_maps = []
        if not self.with_crowd:
            gt_bboxes_ignore = None
        for i, stride in enumerate(self.strides):
            pos_map, scale_map, offset_map = self.calc_gt_center(gt_bboxes, gt_bboxes_ignore, \
                                            stride=stride, regress_range=self.regress_ranges[i], image_shape=ori_shape)
            pos_maps.append(pos_map)
            scale_maps.append(scale_map)
            offset_maps.append(offset_map)


        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_bboxes=DC(to_tensor(gt_bboxes)))
        if self.proposals is not None:
            data['proposals'] = DC(to_tensor(proposals))
        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))
        if self.with_crowd:
            data['gt_bboxes_ignore'] = DC(to_tensor(gt_bboxes_ignore))

        data['classification_maps'] = DC([to_tensor(pos_map) for pos_map in pos_maps])
        data['scale_maps'] = DC([to_tensor(scale_map) for scale_map in scale_maps])
        data['offset_maps'] = DC([to_tensor(offset_map) for offset_map in offset_maps])
        return data

    def calc_gt_center(self, gts, igs, radius=8, stride=4, regress_range=(-1, INF), image_shape=None):

        def gaussian(kernel):
            sigma = ((kernel-1) * 0.5 - 1) * 0.3 + 0.8
            s = 2*(sigma**2)
            dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
            return np.reshape(dx, (-1, 1))
        radius = int(radius/stride)
        if not self.with_width:
            scale_map = np.zeros((2, int(image_shape[0] / stride), int(image_shape[1] / stride)), dtype=np.float32)
        else:
            scale_map = np.zeros((3, int(image_shape[0] / stride), int(image_shape[1] / stride)), dtype=np.float32)
        offset_map = np.zeros((3, int(image_shape[0] / stride), int(image_shape[1] / stride)), dtype=np.float32)
        pos_map = np.zeros((3, int(image_shape[0] / stride), int(image_shape[1] / stride)), dtype=np.float32)
        pos_map[1, :, :, ] = 1  # channel 0: loss weights; channel 1: for ignore, ignore area will be set to 0; channel 2: classification

        if not igs is None and len(igs) > 0:
            igs = igs / stride
            for ind in range(len(igs)):
                x1, y1, x2, y2 = int(igs[ind, 0]), int(igs[ind, 1]), int(np.ceil(igs[ind, 2])), int(np.ceil(igs[ind, 3]))
                pos_map[1, y1:y2, x1:x2] = 0
        half_height = gts[:, 3] - gts[:, 1]
        half_height = (half_height >= regress_range[0]) & (half_height <= regress_range[1])
        inds = half_height.nonzero()
        gts = gts[inds]
        if len(gts) > 0:
            gts = gts / stride
            for ind in range(len(gts)):
                x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])
                c_x, c_y = int((gts[ind, 0] + gts[ind, 2]) / 2), int((gts[ind, 1] + gts[ind, 3]) / 2)

                dx = gaussian(x2-x1)
                dy = gaussian(y2-y1)
                gau_map = np.multiply(dy, np.transpose(dx))

                pos_map[0, y1:y2, x1:x2] = np.maximum(pos_map[0, y1:y2, x1:x2], gau_map)  # gauss map
                pos_map[1, y1:y2, x1:x2] = 1  # 1-mask map
                pos_map[2, c_y, c_x] = 1  # center map

                if not self.with_width:
                    scale_map[0, c_y-radius:c_y+radius+1, c_x-radius:c_x+radius+1] = np.log(gts[ind, 3] - gts[ind, 1])  #value of height
                    scale_map[1, c_y-radius:c_y+radius+1, c_x-radius:c_x+radius+1] = 1  # 1-mask
                else:
                    scale_map[0, c_y-radius:c_y+radius+1, c_x-radius:c_x+radius+1] = np.log(gts[ind, 3] - gts[ind, 1])  #value of height
                    scale_map[1, c_y-radius:c_y+radius+1, c_x-radius:c_x+radius+1] = np.log(gts[ind, 2] - gts[ind, 0])  #value of height
                    scale_map[2, c_y-radius:c_y+radius+1, c_x-radius:c_x+radius+1] = 1  # 1-mask


                offset_map[0, c_y, c_x] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y - 0.5  # height-Y offset
                offset_map[1, c_y, c_x] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x - 0.5  # width-X offset
                offset_map[2, c_y, c_x] = 1  # 1-mask

        return pos_map[None], scale_map[None], offset_map[None]


def _brightness(image, min=0.5, max=2.0):
    '''
    Randomly change the brightness of the input image.
    Protected against overflow.
    '''
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    random_br = np.random.uniform(min, max)

    # To protect against overflow: Calculate a mask for all pixels
    # where adjustment of the brightness would exceed the maximum
    # brightness value and set the value to the maximum at those pixels.
    mask = hsv[:, :, 2] * random_br > 255
    v_channel = np.where(mask, 255, hsv[:, :, 2] * random_br)
    hsv[:, :, 2] = v_channel

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def resize_image(image, gts, igs, scale=(0.4, 1.5)):
    height, width = image.shape[0:2]
    ratio = np.random.uniform(scale[0], scale[1])
    # if len(gts)>0 and np.max(gts[:,3]-gts[:,1])>300:
    #     ratio = np.random.uniform(scale[0], 1.0)
    new_height, new_width = int(ratio * height), int(ratio * width)
    image = cv2.resize(image, (new_width, new_height))
    if len(gts) > 0:
        gts = np.asarray(gts, dtype=float)
        gts[:, 0:4:2] *= ratio
        gts[:, 1:4:2] *= ratio

    if len(igs) > 0:
        igs = np.asarray(igs, dtype=float)
        igs[:, 0:4:2] *= ratio
        igs[:, 1:4:2] *= ratio

    return image, gts, igs


def random_crop(image, gts, gt_labels, igs, crop_size, limit=8):
    img_height, img_width = image.shape[0:2]
    crop_h, crop_w = crop_size

    if len(gts) > 0:
        sel_id = np.random.randint(0, len(gts))
        sel_center_x = int((gts[sel_id, 0] + gts[sel_id, 2]) / 2.0)
        sel_center_y = int((gts[sel_id, 1] + gts[sel_id, 3]) / 2.0)
    else:
        sel_center_x = int(np.random.randint(0, img_width - crop_w + 1) + crop_w * 0.5)
        sel_center_y = int(np.random.randint(0, img_height - crop_h + 1) + crop_h * 0.5)

    crop_x1 = max(sel_center_x - int(crop_w * 0.5), int(0))
    crop_y1 = max(sel_center_y - int(crop_h * 0.5), int(0))
    diff_x = max(crop_x1 + crop_w - img_width, int(0))
    crop_x1 -= diff_x
    diff_y = max(crop_y1 + crop_h - img_height, int(0))
    crop_y1 -= diff_y
    cropped_image = np.copy(image[crop_y1:crop_y1 + crop_h, crop_x1:crop_x1 + crop_w])
    # crop detections
    if len(igs) > 0:
        igs[:, 0:4:2] -= crop_x1
        igs[:, 1:4:2] -= crop_y1
        igs[:, 0:4:2] = np.clip(igs[:, 0:4:2], 0, crop_w)
        igs[:, 1:4:2] = np.clip(igs[:, 1:4:2], 0, crop_h)
        keep_inds = ((igs[:, 2] - igs[:, 0]) >= 8) & \
                    ((igs[:, 3] - igs[:, 1]) >= 8)
        igs = igs[keep_inds]
    if len(gts) > 0:
        ori_gts = np.copy(gts)
        gts[:, 0:4:2] -= crop_x1
        gts[:, 1:4:2] -= crop_y1
        gts[:, 0:4:2] = np.clip(gts[:, 0:4:2], 0, crop_w)
        gts[:, 1:4:2] = np.clip(gts[:, 1:4:2], 0, crop_h)

        before_area = (ori_gts[:, 2] - ori_gts[:, 0]) * (ori_gts[:, 3] - ori_gts[:, 1])
        after_area = (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1])

        keep_inds = ((gts[:, 2] - gts[:, 0]) >= limit) & \
                    (after_area >= 0.5 * before_area)
        gts = gts[keep_inds]
        gt_labels = gt_labels[keep_inds]

    return cropped_image, gts, gt_labels, igs


def random_pave(image, gts, gt_labels, igs, pave_size, limit=8):
    img_height, img_width = image.shape[0:2]
    pave_h, pave_w = pave_size
    # paved_image = np.zeros((pave_h, pave_w, 3), dtype=image.dtype)
    paved_image = np.ones((pave_h, pave_w, 3), dtype=image.dtype) * np.mean(image, dtype=int)
    pave_x = int(np.random.randint(0, pave_w - img_width + 1))
    pave_y = int(np.random.randint(0, pave_h - img_height + 1))
    paved_image[pave_y:pave_y + img_height, pave_x:pave_x + img_width] = image
    # pave detections
    if len(igs) > 0:
        igs[:, 0:4:2] += pave_x
        igs[:, 1:4:2] += pave_y
        keep_inds = ((igs[:, 2] - igs[:, 0]) >= 8) & \
                    ((igs[:, 3] - igs[:, 1]) >= 8)
        igs = igs[keep_inds]

    if len(gts) > 0:
        gts[:, 0:4:2] += pave_x
        gts[:, 1:4:2] += pave_y
        keep_inds = ((gts[:, 2] - gts[:, 0]) >= limit)
        gts = gts[keep_inds]
        gt_labels = gt_labels[keep_inds]

    return paved_image, gts, gt_labels, igs


def augment(img, gt_bboxes, gt_labels, gt_bboxes_ignore, size_train):
    size_train = (size_train[1], size_train[0])
    img_height, img_width = img.shape[:2]
    gt_bboxes[:, [0, 2]] = np.clip(gt_bboxes[:, [0, 2]], 0, img_width-1)
    gt_bboxes[:, [1, 3]] = np.clip(gt_bboxes[:, [1, 3]], 0, img_height-1)
    gt_bboxes_ignore[:, [0, 2]] = np.clip(gt_bboxes_ignore[:, [0, 2]], 0, img_width-1)
    gt_bboxes_ignore[:, [1, 3]] = np.clip(gt_bboxes_ignore[:, [1, 3]], 0, img_height-1)

    # random brightness
    if np.random.randint(0, 2) == 0:
        img = _brightness(img, min=0.5, max=2)
    # random horizontal flip
    if  np.random.randint(0, 2) == 0:
        img = cv2.flip(img, 1)
        if len(gt_bboxes) > 0:
            gt_bboxes[:, [0, 2]] = img_width - gt_bboxes[:, [2, 0]]
        if len(gt_bboxes_ignore) > 0:
            gt_bboxes_ignore[:, [0, 2]] = img_width - gt_bboxes_ignore[:, [2, 0]]

    img, gt_bboxes, gt_bboxes_ignore = resize_image(img, gt_bboxes, gt_bboxes_ignore, scale=(0.4, 1.5))

    if img.shape[0] >= size_train[0]:
        img, gt_bboxes, gt_labels, gt_bboxes_ignore = random_crop(img, gt_bboxes, gt_labels, gt_bboxes_ignore, size_train, limit=16)
    else:
        img, gt_bboxes, gt_labels, gt_bboxes_ignore = random_pave(img, gt_bboxes, gt_labels, gt_bboxes_ignore, size_train, limit=16)

    img_height, img_width = img.shape[:2]
    gt_bboxes[:, [0, 2]] = np.clip(gt_bboxes[:, [0, 2]], 0, img_width - 1)
    gt_bboxes[:, [1, 3]] = np.clip(gt_bboxes[:, [1, 3]], 0, img_height - 1)
    gt_bboxes_ignore[:, [0, 2]] = np.clip(gt_bboxes_ignore[:, [0, 2]], 0, img_width - 1)
    gt_bboxes_ignore[:, [1, 3]] = np.clip(gt_bboxes_ignore[:, [1, 3]], 0, img_height - 1)

    return img, gt_bboxes, gt_labels, gt_bboxes_ignore