from abc import ABCMeta
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torchvision import transforms
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from .builder import DETECTORS
from .builder import build_base_model, build_loss


@DETECTORS.register_module()
class CATDetector(BaseModule, metaclass=ABCMeta):
    def __init__(self,
                 base_model,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.base_model = build_base_model(base_model)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.seg_loss = build_loss(train_cfg.seg_loss)
        self.seg_loss_weights = train_cfg.seg_loss_weights
        self.cls_loss = build_loss(train_cfg.cls_loss)

        self.p_balance_scale = train_cfg.get('p_balance_scale', 0.5)
        self.n_balance_scale = train_cfg.get('n_balance_scale', 0.5)

    def forward_train(self, 
                    img, 
                    img_metas, 
                    gt_semantic_seg, 
                    img_label=None,
                    qtables=None):
        pred_mask, pred_logit = self.base_model(img, qtables)

        gt_semantic_seg = gt_semantic_seg.to(torch.float32)
        gt_semantic_seg = torch.nn.functional.interpolate(gt_semantic_seg, 
                                                        size=pred_mask.shape[2:], 
                                                        mode='bilinear', 
                                                        align_corners=True)
        gt_semantic_seg[gt_semantic_seg > 0.5] = 1
        gt_semantic_seg[gt_semantic_seg <= 0.5] = 0

        mask_balance = torch.ones_like(gt_semantic_seg, dtype=torch.float)
        mask = gt_semantic_seg
        if (mask == 1).sum():
            mask_balance[mask == 1] = self.p_balance_scale / ((mask == 1).sum().to(torch.float) / mask.numel())
            mask_balance[mask == 0] = self.n_balance_scale / ((mask == 0).sum().to(torch.float) / mask.numel())

        losses = dict()
        # seg_loss = self.seg_loss(pred_mask, gt_semantic_seg)
        seg_loss = torch.mean(self.seg_loss(pred_mask, gt_semantic_seg) * mask_balance)
        losses[f'seg_loss'] = seg_loss * self.seg_loss_weights[0]

        cls_loss = self.cls_loss(pred_logit, img_label.squeeze(dim=1))
        losses[f'cls_loss'] = cls_loss * self.seg_loss_weights[1]
        
        return losses

    def forward_test(self, imgs, img_metas, **kwargs):
        return self.simple_test(imgs, img_metas, **kwargs)

    def inference(self, img, img_meta, qtables, rescale):
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)

        seg_logit, cls = self.base_model(img, qtables)

        if img_meta[0]['pad_pre_shape'] is not None \
            and img_meta[0]['pad_shape'] is not None \
            and img_meta[0]['pad_shape'] != img_meta[0]['pad_pre_shape']:
            pad_pre_shape = img_meta[0]['pad_pre_shape'][:2]
            seg_logit = seg_logit[:, :, 0:pad_pre_shape[0], 0:pad_pre_shape[1]]

        if rescale:
            size = img_meta[0]['ori_shape'][:2]
            seg_logit = F.interpolate(seg_logit, size, mode='bilinear', align_corners=True)

        if cls.shape[1] == 1:
            cls = cls.flatten()
            cls_pred = torch.sigmoid(cls)
        else:
            cls_pred = F.softmax(cls, dim=1)
            # 2 classes
            cls_pred = cls_pred[:, 1]
        
        if seg_logit.shape[1] == 1:
            seg_pred = torch.sigmoid(seg_logit)
        else:
            seg_pred = F.softmax(seg_logit, dim=1)
        # seg_pred = seg_logit

        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                seg_pred = seg_pred.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                seg_pred = seg_pred.flip(dims=(2, ))

        return cls_pred, seg_pred

    def simple_test(self, img, img_meta, qtables, rescale=True, **kwargs):
        """Simple test with single image."""
        cls_pred, seg_pred = self.inference(img, img_meta, qtables, rescale)
        cls_pred = cls_pred.cpu().numpy()
        seg_pred = seg_pred.cpu().numpy()

        return cls_pred, seg_pred

    # @auto_fp16(apply_to=('img',))
    @force_fp32(apply_to=('img',))
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
            num_samples=len(data_batch['img_metas']))

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
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        # If the loss_vars has different length, raise assertion error
        # to prevent GPUs from infinite waiting.
        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            dist.all_reduce(log_var_length)
            message = (f'rank {dist.get_rank()}' +
                       f' len(log_vars): {len(log_vars)}' + ' keys: ' +
                       ','.join(log_vars.keys()) + '\n')
            assert log_var_length == len(log_vars) * dist.get_world_size(), \
                'loss log variables are different across GPUs!\n' + message

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars
