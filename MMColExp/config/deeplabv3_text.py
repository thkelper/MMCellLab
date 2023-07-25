work_dir = './records'
# dataset settings
dataset_type = 'MaskSegDatasetv2'
img_norm_cfg = dict(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225], 
                    # to_rgb=False
                    )
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', binary=True, train=True),
    dict(type='SimpleResize', size=(512, 512)),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='Normalize', **img_norm_cfg),
    dict(type='Normalizev2', **img_norm_cfg),
    # dict(type='Pad', size_divisor=16, pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 
                            'gt_semantic_seg', 
                            'edge_gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='SimpleResize', size=(512, 512)),
    # dict(type='Normalize', **img_norm_cfg),
    dict(type='Normalizev2', **img_norm_cfg),
    # dict(type='Pad', size_divisor=16, pad_val=0, seg_pad_val=0),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img']),
]
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3,
    train=dict(
        type=dataset_type,
        data_root = '/mnt/disk1/data/image_forgery/tianchi_text_forgory',
        ann_path='train_w_au.txt',
        edge_mask_dir='/mnt/disk1/data/image_forgery/tianchi_text_forgory/train/edge_mask/',
        pipeline=train_pipeline),
    val=[
        dict(
        type=dataset_type,
        data_root = '/mnt/disk1/data/image_forgery/tianchi_text_forgory',
        ann_path='val.txt',
        test_mode=True,
        pipeline=test_pipeline,
        dataset_name='tianchi_text_forgory',
        ),
    ]
)

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='Detector',
    base_model=dict(
        type='DeepLabV3Plus',
        # encoder_name='resnet34',
        encoder_name='pvt_v2_b2',
        # encoder_name='pvt_v2_b4',
        # pretrined_path='../user_data/pretrained/pvt_v2_b3.pth',
        pretrined_path='/home/yangwu/.cache/torch/checkpoints/pvt_v2_b2.pth',
        # encoder_name = 'timm-efficientnet-b0',
        encoder_weights='imagenet',
        classes=1),
    train_cfg=dict(
        # edge_loss=dict(
        #     type='BinaryDiceLoss',
        #     loss_weight=1.0,
        # ),
        seg_loss=dict(
            type='BinaryDiceLoss',
            loss_weight=1.0,
            class_weight=[1.0, 1.0],
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
                # max_iters=21000)
                max_iters=10000)
                # max_iters=8000)
checkpoint_config = dict(by_epoch=False,
                         interval=1000,
                         max_keep_ckpts=2,
                        )
evaluation = dict(
                interval=500,
                # interval=1,
                 metric='mFscore',
                  pre_eval=True,
                  thresh=0.5,)

log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
find_unused_parameters=True

