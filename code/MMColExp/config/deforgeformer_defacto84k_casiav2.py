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
    samples_per_gpu=9,
    # samples_per_gpu=1,
    # samples_per_gpu=2,
    workers_per_gpu=3,
    # train=dict(
    #     type=dataset_type,
    #     data_root = '/home/yangwu/data/image_forgery/CASIA2',
    #     ann_path='CASIAv2.txt',
    #     # edge_mask_dir='/mnt/disk1/data/image_forgery/CASIA2/edge_mask',
    #     # pipeline=train_pipeline
    #     pipeline=[train_pre_pipeline,
    #               train_post_pipeline],
    #     # simulate_p=0.3,
    #     ),
    train=dict(
            type=dataset_type,
            data_root = '/mnt/disk1/data/image_forgery/defacto',
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
            data_root='/mnt/disk1/data/image_forgery/nist16/nist16',
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
            data_root='/mnt/disk1/data/image_forgery/CASIA1',
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
            data_root='/mnt/disk1/data/image_forgery/IMD20/real_life',
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
            data_root='/mnt/disk1/data/image_forgery/COVERAGE',
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
            data_root = '/mnt/disk1/data/image_forgery/tianchi_text_forgory',
            ann_path='val.txt',
            test_mode=True,
            pipeline=test_pipeline,
            dataset_name='tianchi_text_forgory',
            gt_seg_map_loader_cfg=dict(
                                binary=True, 
                                img_label_binary=True
                                ),
        ),
        dict(
            type=dataset_type,
            data_root='/mnt/disk1/data/image_forgery/defacto',
            ann_path='DEFACTO84k-val.txt',
            test_mode=True,
            pipeline=test_pipeline,
            dataset_name='DEFACTO84k-val',
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
    type='DeforgeFormerDetector',
    base_model=dict(
        # type='DeforgeFormer',
        type='DeforgeFormer3',
        pretrained='/home/yangwu/workspace/image_forgery_detection/code/model_40_clip.pth',
        rgb_backbone=dict(
                type='PVTv2B2', 
                pretrained='/home/yangwu/.cache/torch/hub/checkpoints/pvt_v2_b2.pth',
                # in_chans=15,
                ),
        # srm_conv=True,
        use_rgb=True,
        only_bayar=False,
        # bayar_conv_cfg=dict(
        #     in_channels=3,
        #     out_channels=3,
        #     kernel_size=5,
        #     padding=2,
        # ),    
        # fad_cfg=dict(
        #     size=512,
        # ),
        # f_fusion_mode='element_add',
        f_fusion_mode='concat',
        decoder_cfg=dict(
            out_channels=256,
            atrous_rates=(12, 24, 36),
        ),
        classifier_cfg=dict(
            classes=1,
            pooling="avg", 
            dropout=0.0, 
            activation=None,
        ),
        seg_cfg=dict(
            # out_channels=1,
            out_channels=2,
            kernel_size=3, 
            activation=None,
            upsampling=4,
            # upsampling=32,
        ),
    ),
    train_cfg=dict(
        # seg_loss=dict(
        #     type='BinaryDiceLoss',
        #     loss_weight=1.0,
        # ),
        seg_loss=dict(
            type='DiceLoss',
            loss_weight=1.0,
            class_weight=[1.0, 1.0]
        ),
        clf_loss=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
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
                max_iters=21000)
                # max_iters=42000)
                # max_iters=5)
checkpoint_config = dict(by_epoch=False,
                         interval=1000,
                         max_keep_ckpts=1,
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
