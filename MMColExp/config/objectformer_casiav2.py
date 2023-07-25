work_dir = './records'
# dataset settings
dataset_type = 'MaskSegDatasetv2'
img_norm_cfg = dict(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225], 
                    freq_img_norm=True,
                    )
hpf_size = 10
input_size=(512, 512)

train_pre_pipeline = [
    dict(type='LoadImageFromFile', 
                 gen_high_freq=True,
                 high_pass_filter_size=hpf_size),
    dict(type='LoadAnnotations', binary=True, 
                                train=True,
                                img_label_binary=True),
]

train_post_pipeline = [
    dict(type='SimpleResize', size=input_size),
    # dict(type='SimpleResize', size=(256, 256)),
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
                                'img_label',
                                'high_freq_img']),
]

test_pipeline = [
    dict(type='LoadImageFromFile', gen_high_freq=True,
                                   high_pass_filter_size=hpf_size),
    dict(type='SimpleResize', size=input_size),
    # dict(type='SimpleResize', size=(256, 256)),
    dict(type='Normalizev2', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img', 'high_freq_img']),
    dict(type='Collect', keys=['img', 'high_freq_img']),
]
data = dict(
    # samples_per_gpu=3,
    samples_per_gpu=9,
    workers_per_gpu=3,
    train=dict(
        type=dataset_type,
        data_root = '/home/yangwu/data/image_forgery/CASIA2',
        ann_path='CASIAv2.txt',
        # edge_mask_dir='/mnt/disk1/data/image_forgery/CASIA2/edge_mask',
        # pipeline=train_pipeline
        pipeline=[train_pre_pipeline,
                  train_post_pipeline],
        # simulate_p=0.3,
        ),
    val=[
        dict(
            type=dataset_type,
            data_root='/home/yangwu/data/image_forgery/nist16/nist16_down',
            ann_path='nist16.txt',
            test_mode=True,
            pipeline=test_pipeline,
            dataset_name='nist16',
        ),
        dict(
            type=dataset_type,
            data_root='/home/yangwu/data/image_forgery/CASIA1',
            ann_path='CASIAv1.txt',
            test_mode=True,
            pipeline=test_pipeline,
            dataset_name='CASIA1',
        ),
        dict(
            type=dataset_type,
            data_root='/home/yangwu/data/image_forgery/IMD20/real_life',
            ann_path='IMD20.txt',
            test_mode=True,
            pipeline=test_pipeline,
            dataset_name='IMD20',
        ),
        dict(
            type=dataset_type,
            data_root='/home/yangwu/data/image_forgery/COVERAGE',
            ann_path='COVERAGE.txt',
            test_mode=True,
            pipeline=test_pipeline,
            dataset_name='COVERAGE',
        ),
    ],

)

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='ObjectFormerDetector',
    base_model=dict(
        type='ObjectFormer',
        rgb_backbone=dict(name='efficientnet-b4', 
                          pretrained=True),
        objects_N=16,
        # objects_N=32,
        high_freq_extr_channels=[24, 48, 96, 192, 320],
        embedding_channels=640,
        rgb_embed_extractor=False,
        num_att_layers=8,
        att_dropout=0.0,
        #  att_dropout=0.3,
        drop=0.0,
        # drop=0.3,
        heads=1,
        # heads=4,
        qkv_bias=False,
        bcim=True,
        # win_size=3,
        win_size=9,
        max_mask_size=60,
        bcim_mode='add',
        # custom_init=False,
        custom_init=True,
        # channel_fusion=True,
        channel_fusion=False,
        freq_backbone=dict(
                          name='default',
                        #   name='efficientnet-b4', 
                        #   pretrained=True
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
        seg_loss=dict(
            type='BinaryDiceLoss',
            # type='CrossEntropyLoss',
            # use_sigmoid=True,
            loss_weight=1.0,
        ),
        clf_loss=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=0.2,
            # loss_weight=1.0,
        ),
    ),
    test_cfg=dict()
)

optimizer = dict(type='AdamW',
                 lr=1e-4,
                weight_decay=0.0001)
# optimizer_config = dict()
optimizer_config = dict(
                      grad_clip=dict(
                      max_norm=10.0)
                    )

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
                max_iters=42000)
                # max_iters=5)
checkpoint_config = dict(by_epoch=False,
                         interval=1000,
                         max_keep_ckpts=2,
                        )
evaluation = dict(
                interval=1000,
                # interval=2,
                metric='mFscore',
                pre_eval=True,
                thresh=0.5,)

log_config = dict(
    interval=30,
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

