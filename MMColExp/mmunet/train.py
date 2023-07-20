import torch
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import HOOKS, build_optimizer, build_runner
from mmengine import build_from_cfg
from .datasets import build_dataset, build_dataloader
from .utils.evaluate import EvalHook, DistEvalHook, dataloader_test
from .utils import get_root_logger, find_latest_checkpoint


def train_segmentor(model,
                    dataset,
                    cfg,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None,
                    only_eval=False):
    logger = get_root_logger(cfg.log_level)
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            dist=distributed,
            seed = cfg.seed,
            drop_last=True
        ) for ds in dataset  # 这里竟然还可以这么写
    ]
    # put model on gpus
    if distributed:
        if 'norm_cfg' in cfg and cfg.norm_cfg.type == "SyncBN":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    find_unused_parameters = cfg.get('find_unused_parameters', False)

    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False,
        find_unused_parameters=find_unused_parameters
    )

    # build optimizer 
    optimizer = build_optimizer(model, cfg.optimizer)
    
    if cfg.get('runner') is None:
        cfg.runner = {"type": "IterBasedRunner", 'max_iters': cfg.total_iters}
    
    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta
        )
    )

    # register hooks
    runner.register_training_hooks(cfg.lr_config,
                                   cfg.optimizer_config,
                                   cfg.checkpoint_config,
                                   cfg.log_config,
                                   cfg.get('momentum_config', None))

    runner.timestamp = timestamp

    eval_hooks = []
    eval_dataloaders = []
    if validate:
        vals = cfg.data.val
        if not isinstance(vals, list):
            vals = [vals]
        for val in vals:
            val_dataset = build_dataset(val, dict(test_mode=True))
            val_dataloader = build_dataloader(
                val_dataset,
                samples_per_gpu=1,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=distributed,
                shuffle=False,
            )
            eval_cfg = cfg.get('evaluation', {})
            eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
            eval_hook = DistEvalHook if distributed else EvalHook

            h = eval_hook(val_dataloader, **eval_cfg)
            runner.register_hook(h, priority='LOW')

    # user_defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hoo
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
            'Each item in custom_hooks expects dict type, but got' \
            f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    if cfg.resume_from is None and cfg.get('auto_resume'):
        resume_from = find_latest_checkpoint(cfg.work_dir)
        if resume_from is not None:
            cfg.resume_from = resume_from
        
    if cfg.resume_from:
        runner.load_checkpoint(cfg.load_from)

    if not only_eval:
        runner.run(data_loaders, cfg.workflow)
    else:
        for val_hood in eval_hooks:
            val_hood._do_evaluate(runner)

    if 'test' in cfg.data:
        test = cfg.data.test
        test_dataset = build_dataset(test, dict(test_mode=True))
        test_dataloader = build_dataloader(
            test_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
        )
    