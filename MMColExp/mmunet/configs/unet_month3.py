work_dir = './records'
# dataset settings
dataset_type = 'MMDatasetV2'
img_norm_cfg = dict(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225], 
                    # mean=None, 
                    # std=None, 
                    )
input_size=(512, 512)
# input_size=(224, 224)
# input_size=(288, 288)
# input_size=(384, 384)
# input_size=(448, 448)

train_pre_pipeline = [
    dict(type='LoadImageFromFile',),
    dict(type='LoadAnnotations', binary=True, 
                                train=True,
                               ),
]

train_post_pipeline = [
    dict(type='SimpleResize', size=input_size),

    # dict(type='Pad', size_divisor=32),

    dict(type='RandomFlip', prob=0.5),

    # dict(type='Oneof', 
    #     transforms=[
    #         dict(type='Blur', prob=0.5),
    #         dict(type='PhotoMetricDistortion'),
    #         dict(type='GaussianNoise', p=0.5, mean=0, var=10),
    #         ],
    #     prob=1.0,
    #     ),

    dict(type='Normalizev2', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 
                                'gt_semantic_seg', 
                                ]),
]

test_pipeline = [
    dict(type='LoadImageFromFile', ),

    # dict(type='Pad', size_divisor=32),

    dict(type='SimpleResize', size=input_size),
    dict(type='Normalizev2', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img', ]),
    dict(type='Collect', keys=['img', ]),
]


data = dict(
    # samples_per_gpu=1,
    samples_per_gpu=2,
    workers_per_gpu=1,

    train=dict(
            type=dataset_type,
            # data_root='/home/yangwu/data/image_forgery/CASIA1',
            data_root='/mnt/d/ycp/data/month3',         
            ann_path='train.txt',
            # edge_mask_dir='/mnt/disk1/data/image_forgery/CASIA2/edge_mask',
            # pipeline=train_pipeline
            pipeline=[train_pre_pipeline,
                    train_post_pipeline],
            # simulate_p=0.3,
            ),
    val=[
        dict(
            type=dataset_type,
            data_root='/mnt/d/ycp/data/month3',
            # data_root='/home/yangwu/data/image_forgery/CASIA1',
            # data_root='/mnt/disk1/image_forgery/CASIA1',
            ann_path='test.txt',
            test_mode=True,
            pipeline=test_pipeline,
            dataset_name='month3',
            gt_seg_map_loader_cfg=dict(
                                binary=True, 
                                ),
        ),
    ]
)


# model_settings 
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='NestedUnetDetector',
    base_model=dict(
        type='NestedUNet',
        num_classes=1,
        input_channels=3,
        deep_supervision=True,
        vis_feature=False
    ),
    train_cfg=dict(
        seg_loss=[
            dict(
                # type='BCEDiceLoss',
                type="BinaryDiceLoss",
                loss_weight=1.0,
            ),
            dict(
                type='FocalLoss',
                loss_weight=1.0,
            ),
        ],
        # seg_loss_weights=[0.1, 0.2, 0.3, 0.4],
        # seg_loss_weights=[0.4, 0.3, 0.2, 0.1],
        seg_loss_weights=[0.5, 0.3, 0.2],
    ),
    test_cfg=dict()
)

optimizer = dict(type='AdamW',
                 lr=1e-4,
                weight_decay=0.0001)
optimizer_config = dict()

lr_config = dict(policy='CosineAnnealing',
                 min_lr=1e-7,
                #  warmup='exp',
                #  warmup_iters=1000,
                #  warmup_ratio=0.1,
                by_epoch=False)

checkpoint_config = dict(by_epoch=False,
                        #  interval=1000,
                        # interval=3000,
                        # interval=2200,
                        interval=150,
                        # interval=1500,
                         max_keep_ckpts=1,
                        )
# runtime settings
runner = dict(type='IterBasedRunner',
                # max_iters=21000)
                # max_iters=63000)
                # max_iters=45000)
                max_iters=3000)
                # max_iters=30000)
                # max_iters=5)

evaluation = dict(
                # interval=1500,
                # interval=3000,
                # interval=2200,
                interval=150,
                # interval=1,
                metric='mFscore',
                pre_eval=True,
                mean=False,
                thresh=0.5,
                img_thresh=0.5,
                )

log_config = dict(
    interval=50,
    # interval=1,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])

# ext_test_dataset = ['CASIA1', ]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
find_unused_parameters=False

