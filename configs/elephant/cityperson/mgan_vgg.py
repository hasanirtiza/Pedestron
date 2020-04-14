# model settings
model = dict(
    type='MGAN',
    pretrained='modelzoo://vgg16',
    backbone=dict(
        type='VGG',
        depth=16,
        frozen_stages=1),
    neck=None,
    rpn_head=dict(
        type='RPNHead',
        in_channels=512,
        feat_channels=512,
        anchor_scales=[4., 5.4, 7.2, 9.8, 13.2, 17.9, 24.2, 33.0, 44.1, 59.6, 80.0],
        anchor_ratios=[2.44],
        anchor_strides=[8],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
    ),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=512,
        featmap_strides=[8]),
    mgan_head=dict(
        type='MGANHead'),
    bbox_head=dict(
        type='SharedFCBBoxHead',
        num_fcs=2,
        in_channels=512,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=2,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False,
    )
)
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=12000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.0, nms=dict(type='nms', iou_thr=0.5), max_per_img=100)
)
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'datasets/CityPersons/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'val_gt_for_mmdetction.json',
        img_prefix=data_root + '/leftImg8bit_trainvaltest/leftImg8bit/val_all_in_folder/',
        img_scale=(2048, 1024),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True)
)
# yapf:enable
# runtime settings
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '../work_dirs/mgan_50_65'
load_from = None
resume_from = None
workflow = [('train', 1)]
