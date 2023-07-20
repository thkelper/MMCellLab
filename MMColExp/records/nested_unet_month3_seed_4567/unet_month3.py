work_dir = 'records/nested_unet_month3_seed_4567'
dataset_type = 'MMDatasetV2'
img_norm_cfg = dict(
    mean=[
        0.485,
        0.456,
        0.406,
    ], std=[
        0.229,
        0.224,
        0.225,
    ])
input_size = (
    512,
    512,
)
train_pre_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', binary=True, train=True),
]
train_post_pipeline = [
    dict(type='SimpleResize', size=(
        512,
        512,
    )),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Normalizev2',
        mean=[
            0.485,
            0.456,
            0.406,
        ],
        std=[
            0.229,
            0.224,
            0.225,
        ]),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=[
        'img',
        'gt_semantic_seg',
    ]),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='SimpleResize', size=(
        512,
        512,
    )),
    dict(
        type='Normalizev2',
        mean=[
            0.485,
            0.456,
            0.406,
        ],
        std=[
            0.229,
            0.224,
            0.225,
        ]),
    dict(type='ImageToTensor', keys=[
        'img',
    ]),
    dict(type='Collect', keys=[
        'img',
    ]),
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        type='MMDatasetV2',
        data_root='/mnt/d/ycp/data/month3',
        ann_path='train.txt',
        pipeline=[
            [
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', binary=True, train=True),
            ],
            [
                dict(type='SimpleResize', size=(
                    512,
                    512,
                )),
                dict(type='RandomFlip', prob=0.5),
                dict(
                    type='Normalizev2',
                    mean=[
                        0.485,
                        0.456,
                        0.406,
                    ],
                    std=[
                        0.229,
                        0.224,
                        0.225,
                    ]),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=[
                    'img',
                    'gt_semantic_seg',
                ]),
            ],
        ]),
    val=[
        dict(
            type='MMDatasetV2',
            data_root='/mnt/d/ycp/data/month3',
            ann_path='test.txt',
            test_mode=True,
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='SimpleResize', size=(
                    512,
                    512,
                )),
                dict(
                    type='Normalizev2',
                    mean=[
                        0.485,
                        0.456,
                        0.406,
                    ],
                    std=[
                        0.229,
                        0.224,
                        0.225,
                    ]),
                dict(type='ImageToTensor', keys=[
                    'img',
                ]),
                dict(type='Collect', keys=[
                    'img',
                ]),
            ],
            dataset_name='month3',
            gt_seg_map_loader_cfg=dict(binary=True)),
    ])
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='NestedUnetDetector',
    base_model=dict(
        type='NestedUNet',
        num_classes=1,
        input_channels=3,
        deep_supervision=True,
        vis_feature=False),
    train_cfg=dict(
        seg_loss=[
            dict(type='BinaryDiceLoss', loss_weight=1.0),
            dict(type='FocalLoss', loss_weight=1.0),
        ],
        seg_loss_weights=[
            0.5,
            0.3,
            0.2,
        ]),
    test_cfg=dict())
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict()
lr_config = dict(policy='CosineAnnealing', min_lr=1e-07, by_epoch=False)
checkpoint_config = dict(by_epoch=False, interval=150, max_keep_ckpts=1)
runner = dict(type='IterBasedRunner', max_iters=3000)
evaluation = dict(
    interval=150,
    metric='mFscore',
    pre_eval=True,
    mean=False,
    thresh=0.5,
    img_thresh=0.5)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [
    (
        'train',
        1,
    ),
]
cudnn_benchmark = True
find_unused_parameters = False
auto_resume = False
gpu_ids = range(0, 1)
