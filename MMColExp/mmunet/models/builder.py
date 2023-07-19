from mmcv.utils import Registry, build_from_cfg
from mmcv.cnn import MODELS as MMCV_MODELS

DETECTORS = Registry("detector", parent=MMCV_MODELS)
LOSSES = Registry("loss")
BACKBONES = Registry("backbone")
METRICS = Registry("metric")

def build_loss(cfg, default_args = None):
    loss = build_from_cfg(cfg, LOSSES, default_args)
    return loss

def build_backbone(cfg, default_args = None):
    model = build_from_cfg(cfg, BACKBONES, default_args)
    return model

def build_detector(cfg, default_args = None):
    detector = build_from_cfg(cfg, DETECTORS, default_args)
    return detector

def build_metric(cfg, default_args = None):
    metric = build_from_cfg(cfg, METRICS, default_args)
    return metric

