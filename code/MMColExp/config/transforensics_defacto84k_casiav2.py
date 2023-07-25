work_dir = './records'
# dataset settings
dataset_type = 'MaskSegDatasetv2'
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
                                img_label_binary=True),
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
                                'img_label']),
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
    samples_per_gpu=4,
    workers_per_gpu=4,

    train=dict(
            type=dataset_type,
            # data_root = '/mnt/disk1/data/image_forgery/defacto',
            data_root = '/home/yangwu/data/image_forgery/defacto',
            ann_path='DEFACTO84k+CASIA2-train.txt',
            # edge_mask_dir='/mnt/disk1/data/image_forgery/CASIA2/edge_mask',
            # pipeline=train_pipeline
            pipeline=[train_pre_pipeline,
                    train_post_pipeline],
            # simulate_p=0.3,
            ),
    val=[
        dict(
            type=dataset_type,
            # data_root='/mnt/disk1/data/image_forgery/defacto',
            data_root='/home/yangwu/data/image_forgery/defacto',
            ann_path='DEFACTO84k-val.txt',
            test_mode=True,
            pipeline=test_pipeline,
            dataset_name='DEFACTO84k-val',
            gt_seg_map_loader_cfg=dict(
                                binary=True, 
                                img_label_binary=True
                                ),
        ),
        dict(
            type=dataset_type,
            # data_root='/mnt/disk1/data/image_forgery/nist16/nist16',
            data_root='/home/yangwu/data/image_forgery/nist16/nist16',
            # data_root='/home/yangwu/data/image_forgery/nist16/nist16_down_2',
            ann_path='nist16.txt',
            test_mode=True,
            pipeline=test_pipeline,
            dataset_name='nist16',
            gt_seg_map_loader_cfg=dict(
                                binary=True, 
                                img_label_binary=True
                                ),
        ),
        dict(
            type=dataset_type,
            # data_root='/mnt/disk1/data/image_forgery/CASIA1',
            data_root='/home/yangwu/data/image_forgery/CASIA1',
            ann_path='CASIAv1.txt',
            test_mode=True,
            pipeline=test_pipeline,
            dataset_name='CASIA1',
            gt_seg_map_loader_cfg=dict(
                                binary=True, 
                                img_label_binary=True
                                ),
        ),
        dict(
            type=dataset_type,
            # data_root='/mnt/disk1/data/image_forgery/IMD20/real_life',
            data_root='/home/yangwu/data/image_forgery/IMD20/real_life',
            ann_path='IMD20.txt',
            test_mode=True,
            pipeline=test_pipeline,
            dataset_name='IMD20',
            gt_seg_map_loader_cfg=dict(
                                binary=True, 
                                img_label_binary=True
                                ),
        ),
        dict(
            type=dataset_type,
            # data_root='/mnt/disk1/data/image_forgery/COVERAGE',
            data_root='/home/yangwu/data/image_forgery/COVERAGE',
            ann_path='COVERAGE.txt',
            test_mode=True,
            pipeline=test_pipeline,
            dataset_name='COVERAGE',
            gt_seg_map_loader_cfg=dict(
                                binary=True, 
                                img_label_binary=True
                                ),
        ),
        dict(
            type=dataset_type,
            # data_root = '/mnt/disk1/data/image_forgery/tianchi_text_forgory',
            data_root = '/home/yangwu/data/image_forgery/tianchi_text_forgory',
            ann_path='val.txt',
            test_mode=True,
            pipeline=test_pipeline,
            dataset_name='tianchi_text_forgory',
            gt_seg_map_loader_cfg=dict(
                                binary=True, 
                                img_label_binary=True
                                ),
        ),
    ],
    # test=dict(
    #         type=dataset_type,
    #         data_root = '/home/yangwu/data/image_forgery/defacto',
    #         ann_path='DEFACTO12k-test.txt',
    #         test_mode=True,
    #         pipeline=test_pipeline,
    #         dataset_name='DEFACTO84k-test',
    #         gt_seg_map_loader_cfg=dict(
    #             binary=True, 
    #             img_label_binary=True
    #         )
    #     ),
)

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='TransForensicsDetector',
    base_model=dict(
        type='TransForensics',
        # out_layer=2,
        out_layer=1,
        # f_dims=[256, 512, 1024, 2048],
        f_dims=[64, 128, 320, 512],
        # mlp_dim=2048,
        mlp_dim=512,
        # mlp_dim=1024,
        # depth=6,
        depth=3,
        backbone=dict(
                # name='resnet50', 
                # in_channels=3,
                # depth=5,
                # weights="imagenet",
                type='PVTv2B2', 
                pretrained='/home/yangwu/.cache/torch/hub/checkpoints/pvt_v2_b2.pth',
                ),
        classifier_cfg=dict(
            classes=1,
            pooling="avg", 
            dropout=0.0, 
            activation=None,
        ),
        seg_cfg=dict(
            out_channels=1,
            kernel_size=3, 
            activation=None,
        ),
    ),
    train_cfg=dict(
        seg_loss=[
            dict(
                type='BinaryDiceLoss',
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
        clf_loss=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            # loss_weight=0.2,
        ),
    ),
    test_cfg=dict()
)

optimizer = dict(type='AdamW',
                 lr=1e-4,
                weight_decay=0.0001)
optimizer_config = dict()
# optimizer_config = dict(
#                       grad_clip=dict(
#                       max_norm=10.0)
#                     )

# learning policy
lr_config = dict(policy='CosineAnnealing',
                 min_lr=1e-7,
                #  warmup='exp',
                #  warmup_iters=1000,
                #  warmup_ratio=0.1,
                by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner',
                # max_iters=21000)
                max_iters=90000)
                # max_iters=30000)
                # max_iters=5)
checkpoint_config = dict(by_epoch=False,
                        #  interval=1000,
                        interval=3000,
                        # interval=1500,
                         max_keep_ckpts=1,
                        )
evaluation = dict(
                # interval=1500,
                interval=3000,
                # interval=2,
                metric='mFscore',
                pre_eval=True,
                thresh=0.5,)

log_config = dict(
    interval=50,
    # interval=1,
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
find_unused_parameters=False
