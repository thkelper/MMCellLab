work_dir = './records'
# dataset settings
dataset_type = 'MaskSegDataset'
data_root = '../tianchi_data/data'
img_norm_cfg = dict(
    mean=[217.46, 220.17,221.60], std=[41.48, 40.75, 38.87], to_rgb=True)
crop_size = (640, 640)
train_pre_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations',
         binary=True,
         train=True,
         crop_size=(500, 1000)),
]

train_post_pipeline = [
    # dict(type='JpegCompression',
    #      quality_lower=60,
    #      quality_upper=100,
    #      p=0.5),
    # dict(type='Resize', img_scale=(500, 1000), ratio_range=(0.75, 1.5)),

    # dict(type='TransformPro', erase_prob=0.3, hsv_prob=0.3, albu_prob=0.3),
    dict(type='Resize', img_scale=(500, 1000), ratio_range=(0.8, 1.2)),
    # dict(type='RandomRotate', prob=0.5, degree=20),
    # dict(type='RandomCrop', crop_size=crop_size),
    # dict(type='OpticalDistortion',
    #      distort_limit=0.5,
    #      p=0.5),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),
    # dict(type='GaussianNoise', p=0.5, mean=0, var=10),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32, pad_val=0, seg_pad_val=0),
    # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(
        # type='MultiScaleFlipAug',
        # img_scale=(700, 700),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        # flip=False,
        # transforms=[
        #     dict(type='Resize', keep_ratio=True),
        #     # dict(type='RandomFlip'),
        #     dict(type='Normalize', **img_norm_cfg),
        #     dict(type='Pad', size_divisor=16, pad_val=0, seg_pad_val=0),
        #     dict(type='ImageToTensor', keys=['img']),
        #     dict(type='Collect', keys=['img']),
        # ]
    # )
    # dict(type='Resize',
    #      img_scale=(250, 500),
    #      ratio_range=(1.0, 1.01),
    #      keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32, pad_val=0, seg_pad_val=0),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img']),
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=3,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        # img_dir='train_val/train/img',
        # ann_dir='train_val/train/mask',
        # text_det='train_val/train/text_bboxes.json',
        # img_dir='train_val/train_expanded/img',
        # ann_dir='train_val/train_expanded/mask',
        # img_dir='train/img',
        # ann_dir='train/mask',
        # text_det='train/text_bboxes.json',
        img_dir='train_expanded/img',
        ann_dir='train_expanded/mask',
        text_det='train_expanded/text_bboxes.json',
        pipeline=[train_pre_pipeline,
                  train_post_pipeline],
        # simulate_aug = False,
        simulate_aug = True,
        simulate_p=0.5,
    ),

    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train_val/val/img',
        ann_dir='train_val/val/mask',
        test_mode=True,
        pipeline=test_pipeline),
    )

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='MVSSDetector',
    base_model=dict(
        type='DeepLabV3Plus',
        # encoder_name='resnet34',
        encoder_name='pvt_v2_b3',
        # encoder_name='pvt_v2_b4',
        pretrined_path='../user_data/pretrained/pvt_v2_b3.pth',
        # pretrined_path='/root/.cache/torch/hub/checkpoints/pvt_v2_b4.pth',
        # encoder_name = 'timm-efficientnet-b0',
        encoder_weights='imagenet',
        classes=2),
    train_cfg=dict(
        # edge_loss=dict(
        #     type='BinaryDiceLoss',
        #     loss_weight=1.0,
        # ),
        seg_loss=dict(
            type='DiceLoss',
            loss_weight=3.0,
            class_weight=[1.0, 3.0],
        ),
    ),
    test_cfg=dict()
)

optimizer = dict(type='AdamW',
                 lr=1e-4,
                 weight_decay=0.0001)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='CosineAnnealing',
                 min_lr=1e-6,
                 by_epoch=False)
# lr_config = dict(policy='CosineRestart',
#                  periods=[750 ] * (15000//750),
#                  # restart_weights=[1 ] * (15000//750),
#                  restart_weights=list(0.8**i for i in range(15000//750)),
#                  min_lr=1e-7,
#                  by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner',
              # max_iters=25000)
              max_iters=120000)
checkpoint_config = dict(by_epoch=False,
                         interval=750,)
evaluation = dict(interval=750,
                  # by_epoch=True,
                  metric='IoU+f1',
                  pre_eval=True,
                  thresh=0.5,)

log_config = dict(
    interval=40,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
find_unused_parameters=True

