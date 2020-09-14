from .single_stage import SingleStageDetector
from ..registry import DETECTORS
from mmdet.core import bbox2result

import numpy as np
import cv2

@DETECTORS.register_module
class CSP(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 return_feature_maps=False):
        super(CSP, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained)
        self.return_feature_maps = return_feature_maps
    def show_input_debug(self, img, classification_maps, scale_maps, offset_maps):
        img_numpy = img.cpu().numpy().copy()[0]
        # img_numpy = np.transpose(img_numpy, [1, 2, 0]) * [58.395, 57.12, 57.375] + [123.675, 116.28, 103.53]
        img_numpy = np.transpose(img_numpy, [1, 2, 0]) + [102.9801, 115.9465, 122.7717]
        img_numpy = img_numpy[:, :, ::-1]
        img_numpy = img_numpy.astype(np.uint8)
        strides = [8, 16, 32, 64, 128]
        img_nows = []
        for i, stride in enumerate(strides):
            img_now = img_numpy.copy()
            # cls_numpy = classification_maps[0][i].cpu().numpy().copy()[0][2]
            cls_numpy = classification_maps[0][i].cpu().numpy().copy()[0][:80]
            scale_numpy = scale_maps[0][i].cpu().numpy().copy()[0][0] * stride
            offset_numpy = offset_maps[0][i].cpu().numpy().copy()[0][:2]
            cs, ys, xs = cls_numpy.nonzero()
            print(len(ys))
            for c, x, y in zip(cs, xs, ys):
                cv2.imshow(str(c), classification_maps[0][i].cpu().numpy().copy()[0][80+c])
                realx = x
                realy = y
                height = scale_numpy[y, x]
                realy = realy + 0.5 + offset_numpy[0][y, x]
                realx = realx + 0.5 + offset_numpy[1][y, x]
                realy = realy * stride
                realx = realx * stride
                top_y = int(realy - height/2)
                top_x = int(realx)
                down_y = int(realy + height/2)
                down_x = int(realx)
                top_left = (int(top_x - height * 0.1), int(top_y))
                down_right = (int(down_x + height * 0.1), down_y)
                cv2.rectangle(img_now, top_left, down_right, (255, 255, 5*int(c)), 2)
                img_nows.append(img_now)
            cv2.imshow(str(i) +'img', img_now)
        cv2.waitKey(0)

    def show_input_debug_caltech(self, img, classification_maps, scale_maps, offset_maps):
        for j in range(img.shape[0]):
            img_numpy = img.cpu().numpy().copy()[j]
            img_numpy = np.transpose(img_numpy, [1, 2, 0]) * [58.395, 57.12, 57.375] + [123.675, 116.28, 103.53]
            img_numpy = img_numpy[:, :, ::-1]
            img_numpy = img_numpy.astype(np.uint8)
            strides = [4]
            img_nows = []
            for i, stride in enumerate(strides):
                img_now = img_numpy.copy()
                cls_numpy = classification_maps[j][i].cpu().numpy().copy()[0][2]
                ignore_numpy = classification_maps[j][i].cpu().numpy().copy()[0][1]
                cv2.imshow('ignore', ignore_numpy)
                scale_numpy = scale_maps[j][i].cpu().numpy().copy()[0][0] * stride
                offset_numpy = offset_maps[j][i].cpu().numpy().copy()[0][:2]
                ys, xs = cls_numpy.nonzero()
                print(len(ys))
                for x, y in zip(xs, ys):
                    # cv2.imshow(str(c), classification_maps[j][i].cpu().numpy().copy()[0][c])
                    realx = x
                    realy = y
                    height = scale_numpy[y, x]
                    realy = realy + 0.5 + offset_numpy[0][y, x]
                    realx = realx + 0.5 + offset_numpy[1][y, x]
                    realy = realy * stride
                    realx = realx * stride
                    top_y = int(realy - height/2)
                    top_x = int(realx)
                    down_y = int(realy + height/2)
                    down_x = int(realx)
                    top_left = (int(top_x - height * 0.1), int(top_y))
                    down_right = (int(down_x + height * 0.1), down_y)
                    cv2.rectangle(img_now, top_left, down_right, (255, 255, 125), 2)
                    img_nows.append(img_now)
                cv2.imshow(str(i) +'img', img_now)
            cv2.waitKey(0)

    def show_input_debug_head(self, img, classification_maps, scale_maps, offset_maps):
        for j in range(img.shape[0]):
            img_numpy = img.cpu().numpy().copy()[j]
            img_numpy = np.transpose(img_numpy, [1, 2, 0]) * [58.395, 57.12, 57.375] + [123.675, 116.28, 103.53]
            img_numpy = img_numpy[:, :, ::-1]
            img_numpy = img_numpy.astype(np.uint8)
            strides = [4]
            img_nows = []
            for i, stride in enumerate(strides):
                img_now = img_numpy.copy()
                cls_numpy = classification_maps[j][i].cpu().numpy().copy()[0][2]
                ignore_numpy = classification_maps[j][i].cpu().numpy().copy()[0][1]
                cv2.imshow('ignore', ignore_numpy)
                scale_numpy = scale_maps[j][i].exp().cpu().numpy().copy()[0][0] * stride
                offset_numpy = offset_maps[j][i].cpu().numpy().copy()[0][:2]
                ys, xs = cls_numpy.nonzero()
                for x, y in zip(xs, ys):
                    # cv2.imshow(str(c), classification_maps[j][i].cpu().numpy().copy()[0][c])
                    realx = x
                    realy = y
                    height = scale_numpy[y, x]
                    realy = realy + 0.5 + offset_numpy[0][y, x]
                    realx = realx + 0.5 + offset_numpy[1][y, x]
                    realy = realy * stride
                    realx = realx * stride
                    top_y = int(realy)
                    top_x = int(realx)
                    down_y = int(realy + height)
                    down_x = int(realx)
                    top_left = (int(top_x - height * 0.41/2), int(top_y))
                    down_right = (int(down_x + height * 0.41/2), down_y)
                    cv2.rectangle(img_now, top_left, down_right, (255, 255, 125), 2)
                    img_nows.append(img_now)
                cv2.imshow(str(i) +'img', img_now)
            cv2.waitKey(0)

    def show_mot_input_debug(self, img, classification_maps, scale_maps, offset_maps):
        for j in range(img.shape[0]):
            img_numpy = img.cpu().numpy().copy()[j]
            img_numpy = np.transpose(img_numpy, [1, 2, 0]) * [58.395, 57.12, 57.375] + [123.675, 116.28, 103.53]
            # img_numpy = np.transpose(img_numpy, [1, 2, 0]) + [102.9801, 115.9465, 122.7717]
            img_numpy = img_numpy[:, :, ::-1]
            img_numpy = img_numpy.astype(np.uint8)
            strides = [4]
            img_nows = []
            for i, stride in enumerate(strides):
                img_now = img_numpy.copy()
                # cls_numpy = classification_maps[0][i].cpu().numpy().copy()[0][2]
                cls_numpy = classification_maps[j][i].cpu().numpy().copy()[0][2]
                instance_numpy = classification_maps[j][i].cpu().numpy().copy()[0][3]
                scale_numpy = scale_maps[j][i].cpu().numpy().copy()[0][0] * stride
                offset_numpy = offset_maps[j][i].cpu().numpy().copy()[0][:2]
                ys, xs = cls_numpy.nonzero()
                for x, y in zip(xs, ys):
                    c=0
                    cv2.imshow(str(c), classification_maps[j][i].cpu().numpy().copy()[0][2])
                    realx = x
                    realy = y
                    height = scale_numpy[y, x]
                    realy = realy + 0.5 + offset_numpy[0][y, x]
                    realx = realx + 0.5 + offset_numpy[1][y, x]
                    realy = realy * stride
                    realx = realx * stride
                    top_y = int(realy - height/2)
                    top_x = int(realx)
                    down_y = int(realy + height/2)
                    down_x = int(realx)
                    top_left = (int(top_x - height * 0.1), int(top_y))
                    down_right = (int(down_x + height * 0.1), down_y)
                    cv2.rectangle(img_now, top_left, down_right, (255, 255, 5*int(c)), 2)
                    instance = instance_numpy[y, x]
                    cv2.putText(img_now, str(instance), top_left, cv2.FONT_HERSHEY_COMPLEX, 1, 255)
                    img_nows.append(img_now)
                cv2.imshow(str(i) +'img', img_now)
            cv2.waitKey(0)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      classification_maps=None,
                      scale_maps=None,
                      offset_maps=None):
        # for tracking data which batch is produced by dataset instead of data loader
        if type(img) == list:
            img=img[0]
            img_metas=img_metas[0]
            gt_bboxes=gt_bboxes[0]
            gt_labels=gt_labels[0]
            gt_bboxes_ignore = gt_bboxes_ignore[0]
            classification_maps = classification_maps[0]
            scale_maps = scale_maps[0]
            offset_maps = offset_maps[0]

        x = self.extract_feat(img)
        # self.show_input_debug(img, classification_maps, scale_maps, offset_maps)
        # self.show_input_debug_caltech(img, classification_maps, scale_maps, offset_maps)
        # self.show_mot_input_debug(img, classification_maps, scale_maps, offset_maps)
        # self.show_input_debug_head(img, classification_maps, scale_maps, offset_maps)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, classification_maps, scale_maps, offset_maps, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        if self.return_feature_maps:
            return self.bbox_head.get_bboxes_features(*bbox_inputs)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]

    def foward_features(self, features):
        bbox_list = self.bbox_head.get_bboxes(*features)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]