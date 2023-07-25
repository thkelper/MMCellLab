work_dir = 'records/'
dataset_type = 'MaskSegDatasetv2'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                    )
input_size = (512, 512)

train_pre_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations', 
        binary=True, train=True, 
        img_label_binary=True
    )
]
train_post_pipeline = [
    dict(type='SimpleResize', size=input_size),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Normalizev2',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg', 'img_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='SimpleResize', size=input_size),
    dict(
        type='Normalizev2',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
            type=dataset_type,
            data_root='/mnt/disk1/data/image_forgery/guoshoucai_test/通用场景',
            ann_path='ensemble_train.txt',
            pipeline=[train_pre_pipeline,
                    train_post_pipeline],
            ),
    val=[
        dict(
            type=dataset_type,
            data_root='/mnt/disk1/data/image_forgery/guoshoucai_test/通用场景/CASIA1',
            ann_path='val.txt',
            test_mode=True,
            pipeline=test_pipeline,
            dataset_name='CASIA1',
            gt_seg_map_loader_cfg=dict(
                                binary=True, 
                                img_label_binary=True
                                ),
        ),
        # dict(
        #     type=dataset_type,
        #     data_root='/mnt/disk1/data/image_forgery/guoshoucai_test/通用场景/nist16',
        #     ann_path='val.txt',
        #     test_mode=True,
        #     pipeline=test_pipeline,
        #     dataset_name='nist16',
        #     gt_seg_map_loader_cfg=dict(
        #                         binary=True, 
        #                         img_label_binary=True
        #                         ),
        # ),
        dict(
            type=dataset_type,
            data_root='/mnt/disk1/data/image_forgery/guoshoucai_test/通用场景/IMD20',
            ann_path='val.txt',
            test_mode=True,
            pipeline=test_pipeline,
            dataset_name='IMD20',
            gt_seg_map_loader_cfg=dict(
                                binary=True, 
                                img_label_binary=True
                                ),
        ),
    ],

    test=dict(
            type=dataset_type,
            data_root='/mnt/disk1/data/image_forgery/guoshoucai_test/通用场景/nist16',
            ann_path='val.txt',
            test_mode=True,
            pipeline=test_pipeline,
            dataset_name='nist16',
            gt_seg_map_loader_cfg=dict(
                                binary=True, 
                                img_label_binary=True
                                ),
    ),
)

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    # type='TransForensicsDetector',
    type='TransForensicsDetectorv7',
    base_model=dict(
        type='TransForensics',
        out_layer=1,
        f_dims=[64, 128, 320, 512],
        mlp_dim=512,
        depth=3,
        backbone=dict(
            type='PVTv2B2',
            pretrained=
            '/home/yangwu/.cache/torch/hub/checkpoints/pvt_v2_b2.pth'),
        base_fusion_mode='concat',
        classifier_cfg=dict(
                        # classes=1, 
                        classes=2,
                        pooling='avg', 
                        dropout=0.0, 
                        activation=None
                    ),
        seg_cfg=dict(out_channels=1, 
                    kernel_size=3, 
                    activation=None
                )
        ),
    
    train_cfg=dict(
        seg_loss=[
            dict(type='BinaryDiceLoss',
                reduction='none', 
                loss_weight=1.0),
            dict(type='FocalLoss',
                reduction='none', 
                loss_weight=1.0)
        ],
        seg_loss_weights=[0.5, 0.3, 0.2],
        # clf_loss=dict(
        #     type='CrossEntropyLoss', 
        #     use_sigmoid=True, 
        #     loss_weight=1.0,
        #     )
        clf_loss=dict(
            type='CrossEntropyLossLabelSmooth',
            num_cls=2, 
            epsilon=0.1,
            loss_weight=1.0)
        ),
    test_cfg=dict())

optimizer = dict(type='AdamW', 
                lr=0.0001, 
                weight_decay=0.0001
                )
optimizer_config = dict()

lr_config = dict(policy='CosineAnnealing',
                min_lr=1e-07, 
                by_epoch=False
            )
runner = dict(type='IterBasedRunner', 
            max_iters=6000
        )
checkpoint_config = dict(
                    by_epoch=False, 
                    interval=1000, 
                    max_keep_ckpts=1
                    )
evaluation = dict(
    interval=1000,
    metric='mFscore',
    pre_eval=True,
    mean=False,
    thresh=0.5,
    img_thresh=0.5)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])

ext_test_dataset = ['CASIA1']
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
find_unused_parameters = False
auto_resume = False
gpu_ids = range(0, 4)
