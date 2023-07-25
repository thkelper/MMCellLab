work_dir = './records'
# dataset settings
dataset_type = 'MaskSegDatasetv2'
img_norm_cfg = dict(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225], 
                    # to_rgb=False
                    )

train_pre_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', binary=True, 
                                train=True, 
                                is_gen_edge=True,
                                img_label_binary=True),
]

train_post_pipeline = [
    dict(type='SimpleResize', size=(512, 512)),
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
                                'edge_gt_semantic_seg',
                                'img_label']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='SimpleResize', size=(512, 512)),
    dict(type='Normalizev2', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img']),
]
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=5,
    train=dict(
            type=dataset_type,
            data_root = '/mnt/disk1/data/image_forgery/defacto',
            ann_path='DEFACTO24k+CASIA2-train.txt',
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
)

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='Detector',
    base_model=dict(
        type='MVSSNet',
        backbone='resnet50',
        pretrained_base=True,
        nclass=1,
        sobel=True,
        n_input=3,
        constrain=True),
    train_cfg=dict(
        edge_loss=dict(
            type='BinaryDiceLoss',
            loss_weight=0.8,
            class_weight=[1.0, 1.0],
        ),
        seg_loss=dict(
            # type='DiceLoss',
            type='BinaryDiceLoss',
            loss_weight=0.16,
            class_weight=[1.0, 1.0],
        ),
        clf_loss=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=0.04,
            # class_weight=[1.0, 1.0],
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
                 min_lr=1e-7,
                #  warmup='exp',
                #  warmup_iters=1000,
                #  warmup_ratio=0.1,
                 by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner',
                max_iters=61000)
                # max_iters=1)
checkpoint_config = dict(by_epoch=False,
                         interval=3050,
                         max_keep_ckpts=1,
                        )
evaluation = dict(
                interval=3050,
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
find_unused_parameters=True

