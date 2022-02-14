import logging
import os
import sys
from collections import Counter
from typing import Tuple

import torch
import mmcv
import numpy as np
import cv2
import torch

try:
    import wandb
except:
    wandb = None

def convert_box(tag, boxes, box_labels, class_labels, std, scores=None):
    wandb_box = {}
    if isinstance(std, int):
        std = [std, std]
    if len(std) != 4:
        std = std[::-1] * 2
    boxes = boxes/np.array(std)
    boxes = boxes.tolist()
    box_labels = box_labels.tolist()
    class_labels = {k: class_labels[k] for k in range(len(class_labels))}
    wandb_box["class_labels"] = class_labels
    assert len(boxes) == len(box_labels)
    if scores is not None:
        scores = scores.tolist()
        box_data = [
            dict(
                position=dict(minX=box[0], minY=box[1], maxX=box[2], maxY=box[3]),
                class_id=label,
                scores=dict(cls=scores[i]), box_caption="%s (%.3f)" % (class_labels[label], scores[i])
            )
            for i, (box, label) in enumerate(zip(boxes, box_labels))
        ]
    else:
        box_data = [
            dict(
                position=dict(minX=box[0], minY=box[1], maxX=box[2], maxY=box[3]),
                class_id=label,
            )
            for i, (box, label) in enumerate(zip(boxes, box_labels))
        ]

    wandb_box["box_data"] = box_data
    return {tag: wandb.data_types.BoundingBoxes2D(wandb_box, tag)}


def color_transform(img_tensor, mean, std, to_rgb=False):
    img_np = img_tensor.detach().cpu().numpy().transpose((1, 2, 0)).astype(np.float32)
    #height, width = img_np.shape[:2]
    #img_np = cv2.resize(img_np, (int(width/4), int(height/4)))
    return mmcv.imdenormalize(img_np, mean, std, to_bgr=not to_rgb)


def log_image_with_boxes(
    tag: str,
    image: torch.Tensor,
    bboxes: np.ndarray,
    bbox_tag: str = None,
    labels: np.ndarray = None,
    scores: np.ndarray = None,
    class_names: Tuple[str] = None,
    filename: str = None,
    img_norm_cfg: dict = None,
    backend: str = "auto",
    interval: int = 50,
):
    rank = torch.distributed.get_rank(group=None)
    if rank != 0:
        return
    #_log_counter[key] += 1
    #if not (interval == 1 or _log_counter[key] % interval == 1):
    #    return
    if backend == "auto":
        if wandb is None:
            backend = "file"
        else:
            backend = "wandb"

    if backend == "wandb":
        if wandb is None:
            raise ImportError("wandb is not installed")
        assert (
            wandb.run is not None
        ), "wandb has not been initialized, call `wandb.init` first`"

    elif backend != "file":
        raise TypeError("backend must be file or wandb")

    im_shape = image.shape[1:]
    if filename is None:
        filename = f"{_log_counter[key]}.jpg"
    if bbox_tag is not None:
        bbox_tag = "vis"
    if img_norm_cfg is not None:
        image = color_transform(image, **img_norm_cfg)
    if labels is None:
        labels = np.zeros_like(scores, dtype=np.long)
        class_names = ["foreground"]
    im = {}
    im["data_or_path"] = image
    im["boxes"] = convert_box(
        bbox_tag, bboxes, labels, class_names, scores=scores, std=im_shape
    )
    wandb.log({tag: wandb.Image(**im)}, commit=False)
