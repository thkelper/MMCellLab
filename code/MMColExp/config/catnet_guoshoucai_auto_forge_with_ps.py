work_dir = './records'
# dataset settings
dataset_type = 'MaskSegDatasetv2'
img_norm_cfg = dict(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225], 
                    )
input_size=(512, 512)

train_pre_pipeline = [
    dict(type='LoadImageFromFileDCTMask', 
        crop_size=input_size, 
        grid_crop=True,
        blocks=('RGB', 'DCTvol', 'qtable'),
        read_from_jpeg=True,
        dct_channels=1,
        img_label_binary=True,
        training=True,
        # img_norm_cfg=img_norm_cfg,
    ),
    # dict(type='LoadAnnotations', binary=True, 
    #                             train=True,
    #                             img_label_binary=True),
]

train_post_pipeline = [
    # dict(type='Oneof', 
    #     transforms=[
    #         dict(type='JpegCompression',
    #             quality_lower=40,
    #             quality_upper=100,
    #             p=0.3
    #         ),
    #         dict(type='GaussianNoise', 
    #             p=0.3, 
    #             mean=0, 
    #             var=10),
    #     ],
    #     prob=1.0,
    # ),

    # dict(type='SimpleResize', size=input_size),
    # dict(type='Pad', size_divisor=32),

    # dict(type='RandomFlip', prob=0.5),

    # dict(type='Oneof', 
    #     transforms=[
    #         dict(type='Blur', prob=0.5),
    #         dict(type='PhotoMetricDistortion'),
    #         dict(type='GaussianNoise', p=0.5, mean=0, var=10),
    #         ],
    #     prob=1.0,
    #     ),
    # dict(type='Normalizev2', **img_norm_cfg),
    dict(type='DefaultFormatBundleWoToTensor'),
    dict(type='Collect', keys=['img', 
                                'gt_semantic_seg', 
                                'img_label',
                                'qtables']),
]

test_pipeline = [
    dict(type='LoadImageFromFileDCTMask', 
        crop_size=None,
        grid_crop=True,
        blocks=('RGB', 'DCTvol', 'qtable'),
        read_from_jpeg=True,
        dct_channels=1,
        img_label_binary=True,
        training=False,
        pad_size=32,
    ),
    # dict(type='Pad', size_divisor=32),
    # dict(type='SimpleResize', size=input_size),
    # dict(type='Normalizev2', **img_norm_cfg),
    # dict(type='ImageToTensor', keys=['img', ]),
    dict(type='Collect', keys=['img', 'qtables']),
]
data = dict(
    # samples_per_gpu=10,
    samples_per_gpu=5,
    # samples_per_gpu=1,
    workers_per_gpu=5,

    # train=dict(
    #         type=dataset_type,
    #         data_root='/mnt/disk1/data/image_forgery/guoshoucai_auto_gen/train_forged_with_ps',
    #         ann_path='train.txt',
    #         pipeline=[train_pre_pipeline,
    #                 train_post_pipeline],
    #         ),
    train=dict(
            type=dataset_type,
            data_root='/mnt/disk1/data/image_forgery/',
            ann_path='guoshoucai_auto_gen_ps_with_tianchi.txt',
            # ann_path='guoshoucai_auto_gen_ps_with_tianchi_au_val.txt',
            pipeline=[train_pre_pipeline,
                    train_post_pipeline],
            ),
    val=[
        dict(
            type=dataset_type,
            data_root='/mnt/disk1/data/image_forgery/tianchi_text_forgory',
            # data_root='/home/yangwu/data/image_forgery/defacto',
            # data_root='/mnt/disk1/image_forgery/defacto',
            ann_path='val.txt',
            test_mode=True,
            pipeline=test_pipeline,
            dataset_name='tianchi',
            gt_seg_map_loader_cfg=dict(
                                binary=True, 
                                img_label_binary=True
                                ),
        ),
        dict(
            type=dataset_type,
            data_root='/mnt/disk1/data/image_forgery/guoshoucai_auto_gen/test_forged_with_ps',
            # data_root='/home/yangwu/data/image_forgery/defacto',
            # data_root='/mnt/disk1/image_forgery/defacto',
            ann_path='test.txt',
            test_mode=True,
            pipeline=test_pipeline,
            dataset_name='guoshoucai',
            gt_seg_map_loader_cfg=dict(
                                binary=True, 
                                img_label_binary=True
                                ),
        ),
    ]
)

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='CATDetector',
    base_model=dict(
        type='CAT_Net_ORI',
        # crop_size=input_size, 
        pretrained='/home/yangwu/.cache/torch/checkpoints/hrnetv2_w48_imagenet_pretrained.pth',
        dct_pretrained='/home/yangwu/.cache/torch/checkpoints/DCT_djpeg.pth.tar',
        reshape=False,
        fuse_mode='concat',
        skip_dct=False,

        extra=dict(
                FINAL_CONV_KERNEL=1,
                STAGE1=dict(
                    NUM_MODULES= 1,
                    NUM_BRANCHES= 1,
                    BLOCK='BOTTLENECK',
                    NUM_BLOCKS=(4, ),
                    NUM_CHANNELS=(64, ),
                    FUSE_METHOD='SUM',
                ),
                STAGE2=dict(
                    NUM_MODULES= 1,
                    NUM_BRANCHES= 2,
                    BLOCK='BASIC',
                    NUM_BLOCKS=(4, 4),
                    NUM_CHANNELS=(48, 96),
                    FUSE_METHOD='SUM',
                ),
                STAGE3=dict(
                    NUM_MODULES= 4,
                    NUM_BRANCHES= 3,
                    BLOCK='BASIC',
                    NUM_BLOCKS=(4, 4, 4),
                    NUM_CHANNELS=(48, 96, 192),
                    FUSE_METHOD='SUM',
                ),
                STAGE4=dict(
                    NUM_MODULES= 3,
                    NUM_BRANCHES= 4,
                    BLOCK='BASIC',
                    NUM_BLOCKS=(4, 4, 4, 4),
                    NUM_CHANNELS=(48, 96, 192, 384),
                    FUSE_METHOD='SUM',
                ),
                DC_STAGE3=dict(
                    NUM_MODULES= 3,  # 4
                    NUM_BRANCHES= 2,
                    BLOCK='BASIC',
                    NUM_BLOCKS=(4, 4),
                    NUM_CHANNELS=(96, 192),
                    FUSE_METHOD='SUM',
                ),
                DC_STAGE4=dict(
                    NUM_MODULES=2,               NUM_BRANCHES= 3,
                    BLOCK='BASIC',
                    NUM_BLOCKS=(4, 4, 4),
                    NUM_CHANNELS=(96, 192, 384),
                    FUSE_METHOD='SUM',
                ),
                STAGE5=dict(
                    NUM_MODULES=1,               NUM_BRANCHES= 4,
                    BLOCK='BASIC',
                    NUM_BLOCKS=(4, 4, 4, 4),
                    NUM_CHANNELS=(24, 48, 96, 192),
                    FUSE_METHOD='SUM',
                ),
        ),
        cls_head_cfg=dict(
            classes=2,
            pooling="avg", 
            # pooling="max", 
            dropout=0.0, 
            activation=None,
        ),
    ),
    train_cfg=dict(
        # seg_loss=dict(
        #         type='CrossEntropyLoss',
        #         use_sigmoid=True,
        #         # class_weight=(10.0,),
        # ),
        seg_loss=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='none'
        ),
        seg_loss_weights=(5.0, 1.0),
        cls_loss=dict(
            type='CrossEntropyLoss',
            # use_sigmoid=True,
            use_sigmoid=False,
            class_weight=(1.0, 1.0),
        ),
        p_balance_scale=0.05,
        n_balance_scale=1.0,
    ),

    test_cfg=dict()
)

optimizer = dict(type='Adam',
                 lr=1e-4,
                weight_decay=0.00001)
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
                #  max_iters=60980)
                max_iters=30160,)
checkpoint_config = dict(by_epoch=False,
                        # interval=4065,
                        # interval=1508,
                        interval=3016,
                         max_keep_ckpts=1,
                        )
evaluation = dict(
                # interval=4065,
                # interval=1508,
                interval=3016,
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

ext_test_dataset = ['CASIA1', ]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
find_unused_parameters=False
