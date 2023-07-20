from abc import ABCMeta
from collections import OrderedDict
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torchvision import transforms
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from .builder import DETECTORS, SEGMENTORS
from .builder import build_base_model, build_loss

# @DETECTORS.register_module()
@SEGMENTORS.register_module()
class NestedUnetDetector(BaseModule, metaclass=ABCMeta):
    def __init__(self, 
                 base_model,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.base_model = build_base_model(base_model)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.seg_loss_weights = train_cfg.seg_loss_weights
        self.seg_losses = (build_loss(train_cfg.seg_loss[0]),
                            build_loss(train_cfg.seg_loss[1]))
    
        self.clf_loss = None
        if 'clf_loss' in train_cfg:
            self.clf_loss = build_loss(train_cfg.cls_loss)
        
    def compute_seg_loss(self, loss_func1, loss_func2, features, gt_mask):
        return loss_func1(features, gt_mask) + loss_func2(features, gt_mask)
    
    def compute_cls_loss(self, cls_pred, gt_label):
        return self.clf_loss(cls_pred, gt_label)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      img_label=None):
        losses = dict()

        outs = self.base_model(img)
        seg_loss = 0.0
        for i, pred_mask in enumerate(outs):
            seg_loss = self.compute_seg_loss(*self.seg_losses, pred_mask, gt_semantic_seg)
            losses[f'seg_loss_{i}'] = seg_loss * self.seg_loss_weights[i]

        return losses


    def inference(self, img, img_meta, rescale, argmax=False):
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['orishape'] == ori_shape for _ in img_meta)
        seg_logit = self.base_model(img)[0]

        if img_meta[0]['pred_pre_shape'] is not None \
            and img_meta[0]['pad_shape'] is not None \
            and img_meta[0]['pad_shape'] != img_meta[0]['pad_pre_shape']:
            pad_pre_shape = img_meta[0]['pad_pre_shape'][:2]
            seg_logit = seg_logit[:, :, 0:pad_pre_shape[0], 0:pad_pre_shape[1]]
        
        if rescale:
            size = img_meta[0]['ori_shape'][:2]
            seg_logit = F.interpolate(seg_logit, size, mode='bilinear', align_corners=True)
        
        if seg_logit.shape[1] == 1:
            seg_pred = torch.sigmoid(seg_logit)
        else:
            seg_pred = F.softmax(seg_logit, dim=1)
            if argmax:
                seg_pred = seg_pred.argmax(dim=1, keepdim=False)

        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                seg_pred = seg_pred.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                seg_pred = seg_pred.flip(dims=(2, )) 

        return seg_pred

    def simple_test(self, img, img_meta, rescale=True, argmax=False):
        seg_pred = self.inference(img, img_meta, rescale, argmax=argmax)
        seg_pred = seg_pred.cpu().numpy()
        return seg_pred

    def forward_test(self, imgs, img_metas, **kwargs):
        return self.simple_test(imgs, img_metas, **kwargs)
        
    @force_fp32(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)
    
    def train_step(self, data_batch, *args, **kwargs):
        losses = self(**data_batch)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data_batch['img_metas'])
        )
        return outputs


    @staticmethod
    def _parse_losses(losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors'
                )
            
        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)
        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            dist.all_reduce(log_var_length)
            message = (f'rank {dist.get_rank()}' + 
                       f' len(log_vars): {len(log_vars)}' + 'keys:'
                         + ','.join(log_vars.keys()) + '\n')
            assert log_var_length == len(log_vars) * dist.get_world_size(), \
                    'loss log variables are different across GPUS!\n' + message
        
            log_vars['loss'] = loss
            for loss_name, loss_value in log_vars.items():
                if dist.is_available() and dist.is_initialized():
                    loss_value = loss_value.data.clone()
                    dist.all_reduce(loss_value.div_(dist.get_world_size()))
                log_vars[loss_name] = loss_value.item()
            
            return loss, log_vars