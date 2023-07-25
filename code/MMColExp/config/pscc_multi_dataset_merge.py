work_dir = './records'
# dataset settings
dataset_type = 'MaskSegDatasetv2'
img_norm_cfg = dict(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225], 
                    )
input_size=(512, 512)

train_pre_pipeline = [
    dict(type='LoadImageFromFile',),
    dict(type='LoadAnnotations', binary=True, 
                                train=True,
                                img_label_binary=True),
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
    samples_per_gpu=1,
    # samples_per_gpu=2,
    workers_per_gpu=4,

    # train=dict(
    #         type=dataset_type,
    #         data_root='/mnt/disk1/data/image_forgery/guoshoucai_auto_gen/train_forged_with_ps',
    #         ann_path='train.txt',
    #         pipeline=[train_pre_pipeline,
    #                 train_post_pipeline],
    #         ),
    train=dict(
            type=dataset_type,
            # data_root='/mnt/disk1/data/image_forgery/',
            # ann_path='guoshoucai_auto_gen_ps_with_tianchi.txt',
            # ann_path='guoshoucai_auto_gen_ps_with_tianchi_au_val.txt',
            data_root='/mnt/disk1/data/image_forgery/',         
            ann_path='multi_dataset_merged_2.txt',
            pipeline=[train_pre_pipeline,
                    train_post_pipeline],
            ),
    val=[
        # dict(
        #     type=dataset_type,
        #     data_root='/mnt/disk1/data/image_forgery/guoshoucai_auto_gen/test_forged_with_ps',
        #     # data_root='/home/yangwu/data/image_forgery/defacto',
        #     # data_root='/mnt/disk1/image_forgery/defacto',
        #     ann_path='test.txt',
        #     test_mode=True,
        #     pipeline=test_pipeline,
        #     dataset_name='guoshoucai',
        #     gt_seg_map_loader_cfg=dict(
        #                         binary=True, 
        #                         img_label_binary=True
        #                         ),
        # ),
        # dict(
        #     type=dataset_type,
        #     data_root='/mnt/disk1/data/image_forgery/tianchi_text_forgory',
        #     # data_root='/home/yangwu/data/image_forgery/defacto',
        #     # data_root='/mnt/disk1/image_forgery/defacto',
        #     ann_path='val.txt',
        #     test_mode=True,
        #     pipeline=test_pipeline,
        #     dataset_name='tianchi',
        #     gt_seg_map_loader_cfg=dict(
        #                         binary=True, 
        #                         img_label_binary=True
        #                         ),
        # ),
        dict(
            type=dataset_type,
            data_root='/mnt/disk1/data/image_forgery/defacto',
            # data_root='/home/yangwu/data/image_forgery/defacto',
            # data_root='/mnt/disk1/image_forgery/defacto',
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
            data_root='/mnt/disk1/data/image_forgery/PSCCNet',
            # data_root='/home/yangwu/data/image_forgery/defacto',
            # data_root='/mnt/disk1/image_forgery/defacto',
            ann_path='test.txt',
            test_mode=True,
            pipeline=test_pipeline,
            dataset_name='PSCCNet',
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
    type='PSCCDetector',
    base_model=dict(
        type='PSCCNet',
        crop_size=input_size, 
        pretrained='/home/yangwu/.cache/torch/checkpoints/hrnet_w18_small_v2.pth'
    ),
    train_cfg=dict(
        seg_loss=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                reduction='none'
            ),
        seg_loss_weights=[1.0, 1.0],
        cls_loss=dict(
            type='CrossEntropyLoss',
            # use_sigmoid=True,
            use_sigmoid=False,
            class_weight=[4.0, 1.0],
        ),
    ),

    test_cfg=dict()
)

optimizer = dict(type='Adam',
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
                # max_iters=34000)
                # max_iters=38000)
                 max_iters=60980)
checkpoint_config = dict(by_epoch=False,
                        # interval=3400,
                        # interval=3800,
                        interval=4065,
                         max_keep_ckpts=1,
                        )
evaluation = dict(
                # interval=3400,
                # interval=3800,
                interval=4065,
                # interval=50,
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
