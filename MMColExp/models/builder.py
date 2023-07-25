import warnings
from mmcv.utils import Registry, build_from_cfg
from mmcv.cnn import MODELS as MMCV_MODELS

DETECTORS = Registry('detector', parent=MMCV_MODELS)
BASE_MODELS = Registry('base_model')
BACKBONES = Registry('backbones')
LOSSES = Registry('loss')
SEGMENTORS = Registry("segmentor")


def build_base_model(cfg, default_args=None):
    model = build_from_cfg(cfg, BASE_MODELS, default_args)

    return model


def build_loss(cfg, default_args=None):
    loss = build_from_cfg(cfg, LOSSES, default_args)

    return loss


def build_backbone(cfg, default_args=None):
    model = build_from_cfg(cfg, BACKBONES, default_args)

    return model


def build_detector(cfg, default_args=None):
    """Build detector."""
    # if train_cfg is not None or test_cfg is not None:
    #     warnings.warn(
    #         'train_cfg and test_cfg is deprecated, '
    #         'please specify them in model', UserWarning)
    # assert cfg.get('train_cfg') is None or train_cfg is None, \
    #     'train_cfg specified in both outer field and model field '
    # assert cfg.get('test_cfg') is None or test_cfg is None, \
    #     'test_cfg specified in both outer field and model field '
    # return build_from_cfg(cfg, DETECTORS, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))

    return build_from_cfg(cfg, DETECTORS, default_args)


def build_segmentor(cfg, default_args=None):
    segmentor = build_from_cfg(cfg, SEGMENTORS, default_args=default_args)
    return segmentor