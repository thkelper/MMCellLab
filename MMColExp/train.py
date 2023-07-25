import torch
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import HOOKS, build_optimizer, build_runner
from mmcv.utils import build_from_cfg
from .datasets import build_dataset, build_dataloader
from .utils import get_root_logger, find_latest_checkpoint
from .utils.evaluate import EvalHook, DistEvalHook, dataloader_test


def train_segmentor(model,
                    dataset,
                    cfg,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None,
                    only_eval=False):
    """Launch segmentor training."""
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            # len(cfg.gpu_ids),
            # 1,
            dist=distributed,
            seed=cfg.seed,
            drop_last=True) for ds in dataset
    ]

    # put model on gpus
    if distributed:
        if 'norm_cfg' in cfg and cfg.norm_cfg.type == 'SyncBN':
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if cfg.get('runner') is None:
        cfg.runner = {'type': 'IterBasedRunner', 'max_iters': cfg.total_iters}

    runner = build_runner(
                    cfg.runner,
                    default_args=dict(
                        model=model,
                        batch_processor=None,
                        optimizer=optimizer,
                        work_dir=cfg.work_dir,
                        logger=logger,
                        meta=meta))

    # register hooks
    runner.register_training_hooks(cfg.lr_config,
                                   cfg.optimizer_config,
                                   cfg.checkpoint_config,
                                   cfg.log_config,
                                   cfg.get('momentum_config', None))

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    eval_hooks = []
    eval_dataloaders = []
    # register eval hooks
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
                shuffle=False)
            eval_cfg = cfg.get('evaluation', {})
            eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
            eval_hook = DistEvalHook if distributed else EvalHook
            # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
            # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.

            h = eval_hook(val_dataloader, **eval_cfg)
            runner.register_hook(h, priority='LOW')

            eval_hooks.append(h)
            eval_dataloaders.append(val_dataloader)

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    if cfg.resume_from is None and cfg.get('auto_resume'):
        resume_from = find_latest_checkpoint(cfg.work_dir)
        if resume_from is not None:
            cfg.resume_from = resume_from
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
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
            shuffle=False)
        test_cfg = cfg.get('evaluation', {})
        test_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        test_hook = DistEvalHook if distributed else EvalHook
        test_cfg['interval'] = 1
        test_hook(test_dataloader, **test_cfg)._do_evaluate(runner)