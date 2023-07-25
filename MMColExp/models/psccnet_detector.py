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
class PSCCDetector(BaseModule, metaclass=ABCMeta):  
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
        self.mask_loss_weights = train_cfg.get('mask_loss_weights', (1.0, 1.0, 1.0, 1.0))
        self.cls_loss = build_loss(train_cfg.cls_loss)
        self.p_balance_scale = train_cfg.get('p_balance_scale', 0.5)
        self.n_balance_scale = train_cfg.get('n_balance_scale', 0.5)

    def generate_4masks(self, mask):
        h, w = mask.size()[2:]
        mask_tmp = mask.to(torch.float32)
        # print(mask)
        mask2 = torch.nn.functional.interpolate(mask_tmp, size=(h//2, w//2), mode='bilinear', align_corners=True)
        mask2[mask2 > 0.5] = 1
        mask2[mask2 <= 0.5] = 0
        # mask2 = mask2.to(torch.int64)

        mask3 = torch.nn.functional.interpolate(mask_tmp, size=(h//4, w//4), mode='bilinear', align_corners=True)
        mask3[mask3 > 0.5] = 1
        mask3[mask3 <= 0.5] = 0
        # mask3 = mask3.to(torch.int64)

        mask4 = torch.nn.functional.interpolate(mask_tmp, size=(h//8, w//8), mode='bilinear', align_corners=True)
        mask4[mask4 > 0.5] = 1
        mask4[mask4 <= 0.5] = 0
        # mask4 = mask4.to(torch.int64)

        # return mask, mask2, mask3, mask4
        return mask_tmp, mask2, mask3, mask4

    def forward_train(self, 
                    img, 
                    img_metas, 
                    gt_semantic_seg, 
                    img_label=None):
        losses = dict()

        mask1, mask2, mask3, mask4 = self.generate_4masks(gt_semantic_seg)
        # median-frequency class weighting
        mask1_balance = torch.ones_like(mask1, dtype=torch.float)
        p_balance_scale = self.p_balance_scale
        n_balance_scale = self.n_balance_scale
        if (mask1 == 1).sum():
            mask1_balance[mask1 == 1] = p_balance_scale / ((mask1 == 1).sum().to(torch.float) / mask1.numel())
            mask1_balance[mask1 == 0] = n_balance_scale / ((mask1 == 0).sum().to(torch.float) / mask1.numel())
        # else:
        #     print('Mask1 balance is not working!')

        mask2_balance = torch.ones_like(mask2, dtype=torch.float)
        if (mask2 == 1).sum():
            mask2_balance[mask2 == 1] = p_balance_scale / ((mask2 == 1).sum().to(torch.float) / mask2.numel())
            mask2_balance[mask2 == 0] = n_balance_scale / ((mask2 == 0).sum().to(torch.float) / mask2.numel())
        # else:
        #     print('Mask2 balance is not working!')

        mask3_balance = torch.ones_like(mask3, dtype=torch.float)
        if (mask3 == 1).sum():
            mask3_balance[mask3 == 1] = p_balance_scale / ((mask3 == 1).sum().to(torch.float) / mask3.numel())
            mask3_balance[mask3 == 0] = n_balance_scale / ((mask3 == 0).sum().to(torch.float) / mask3.numel())
        # else:
        #     print('Mask3 balance is not working!')

        mask4_balance = torch.ones_like(mask4, dtype=torch.float)
        if (mask4 == 1).sum():
            mask4_balance[mask4 == 1] = p_balance_scale / ((mask4 == 1).sum().to(torch.float) / mask4.numel())
            mask4_balance[mask4 == 0] = n_balance_scale / ((mask4 == 0).sum().to(torch.float) / mask4.numel())
        # else:
        #     print('Mask4 balance is not working!')
        
        (pred_mask1, pred_mask2, pred_mask3, pred_mask4), pred_logit = self.base_model(img)

        # mask1, mask2, mask3, mask4 = mask1.squeeze(dim=1), mask2.squeeze(
        #         dim=1), mask3.squeeze(dim=1), mask4.squeeze(dim=1)

        mask1_loss = torch.mean(self.seg_loss(pred_mask1, mask1) * mask1_balance)
        mask2_loss = torch.mean(self.seg_loss(pred_mask2, mask2) * mask2_balance)
        mask3_loss = torch.mean(self.seg_loss(pred_mask3, mask3) * mask3_balance)
        mask4_loss = torch.mean(self.seg_loss(pred_mask4, mask4) * mask4_balance)
        m1_w, m2_w, m3_w, m4_w = self.mask_loss_weights
        seg_loss = mask1_loss*m1_w + mask2_loss*m2_w + mask3_loss*m3_w + mask4_loss*m4_w
        losses[f'seg_loss'] = seg_loss * self.seg_loss_weights[0]

        cls_loss = self.cls_loss(pred_logit, img_label.squeeze(dim=1))
        losses[f'cls_loss'] = cls_loss * self.seg_loss_weights[1]
        
        return losses

    def forward_test(self, imgs, img_metas, **kwargs):
        return self.simple_test(imgs, img_metas, **kwargs)

    def inference(self, img, img_meta, rescale, argmax=False):
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)

        masks, cls = self.base_model(img)
        seg_logit = masks[0]

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
            if argmax:
                cls_pred = cls_pred.argmax(dim=1, keepdim=False)
            else:
                # 2 classes
                cls_pred = cls_pred[:, 1]
        
        # if seg_logit.shape[1] == 1:
        #     seg_pred = torch.sigmoid(seg_logit)
        # else:
        #     seg_pred = F.softmax(seg_logit, dim=1)
        #     if argmax:
        #         seg_pred = seg_pred.argmax(dim=1, keepdim=False)
        seg_pred = seg_logit

        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                seg_pred = seg_pred.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                seg_pred = seg_pred.flip(dims=(2, ))

        return cls_pred, seg_pred

    def simple_test(self, img, img_meta, rescale=True, argmax=False):
        """Simple test with single image."""
        cls_pred, seg_pred = self.inference(img, img_meta, rescale, argmax=argmax)
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


@DETECTORS.register_module()
class PSCCDetectorv2(BaseModule, metaclass=ABCMeta):
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
        self.aux_seg_losses = (build_loss(train_cfg.aux_seg_loss[0]), build_loss(train_cfg.aux_seg_loss[1]))
        self.aux_mask_loss_weights = train_cfg.get('aux_mask_loss_weights', (1.0, 1.0, 1.0, 1.0))
        self.seg_loss_weights = train_cfg.seg_loss_weights
        self.mask_loss_weights = train_cfg.get('mask_loss_weights', (1.0, 1.0, 1.0, 1.0))
        self.cls_loss = build_loss(train_cfg.cls_loss)

    def generate_4masks(self, mask):
        h, w = mask.size()[2:]
        mask_tmp = mask.to(torch.float32)
        # print(mask)
        mask2 = torch.nn.functional.interpolate(mask_tmp, size=(h//2, w//2), mode='bilinear', align_corners=True)
        mask2[mask2 > 0.5] = 1
        mask2[mask2 <= 0.5] = 0
        # mask2 = mask2.to(torch.int64)

        mask3 = torch.nn.functional.interpolate(mask_tmp, size=(h//4, w//4), mode='bilinear', align_corners=True)
        mask3[mask3 > 0.5] = 1
        mask3[mask3 <= 0.5] = 0
        # mask3 = mask3.to(torch.int64)

        mask4 = torch.nn.functional.interpolate(mask_tmp, size=(h//8, w//8), mode='bilinear', align_corners=True)
        mask4[mask4 > 0.5] = 1
        mask4[mask4 <= 0.5] = 0
        # mask4 = mask4.to(torch.int64)

        # return mask, mask2, mask3, mask4
        return mask_tmp, mask2, mask3, mask4

    def forward_train(self, 
                    img, 
                    img_metas, 
                    gt_semantic_seg, 
                    img_label=None):
        losses = dict()

        mask1, mask2, mask3, mask4 = self.generate_4masks(gt_semantic_seg)
        # median-frequency class weighting
        mask1_balance = torch.ones_like(mask1, dtype=torch.float)
        if (mask1 == 1).sum():
            mask1_balance[mask1 == 1] = 0.5 / ((mask1 == 1).sum().to(torch.float) / mask1.numel())
            mask1_balance[mask1 == 0] = 0.5 / ((mask1 == 0).sum().to(torch.float) / mask1.numel())
        # else:
        #     print('Mask1 balance is not working!')

        mask2_balance = torch.ones_like(mask2, dtype=torch.float)
        if (mask2 == 1).sum():
            mask2_balance[mask2 == 1] = 0.5 / ((mask2 == 1).sum().to(torch.float) / mask2.numel())
            mask2_balance[mask2 == 0] = 0.5 / ((mask2 == 0).sum().to(torch.float) / mask2.numel())
        # else:
        #     print('Mask2 balance is not working!')

        mask3_balance = torch.ones_like(mask3, dtype=torch.float)
        if (mask3 == 1).sum():
            mask3_balance[mask3 == 1] = 0.5 / ((mask3 == 1).sum().to(torch.float) / mask3.numel())
            mask3_balance[mask3 == 0] = 0.5 / ((mask3 == 0).sum().to(torch.float) / mask3.numel())
        # else:
        #     print('Mask3 balance is not working!')

        mask4_balance = torch.ones_like(mask4, dtype=torch.float)
        if (mask4 == 1).sum():
            mask4_balance[mask4 == 1] = 0.5 / ((mask4 == 1).sum().to(torch.float) / mask4.numel())
            mask4_balance[mask4 == 0] = 0.5 / ((mask4 == 0).sum().to(torch.float) / mask4.numel())
        # else:
        #     print('Mask4 balance is not working!')
        
        (pred_mask1, pred_mask2, pred_mask3, pred_mask4), aux_pred_masks, pred_logit = self.base_model(img)

        # mask1, mask2, mask3, mask4 = mask1.squeeze(dim=1), mask2.squeeze(
        #         dim=1), mask3.squeeze(dim=1), mask4.squeeze(dim=1)

        mask1_loss = torch.mean(self.seg_loss(pred_mask1, mask1) * mask1_balance)
        mask2_loss = torch.mean(self.seg_loss(pred_mask2, mask2) * mask2_balance)
        mask3_loss = torch.mean(self.seg_loss(pred_mask3, mask3) * mask3_balance)
        mask4_loss = torch.mean(self.seg_loss(pred_mask4, mask4) * mask4_balance)
        m1_w, m2_w, m3_w, m4_w = self.mask_loss_weights
        seg_loss = mask1_loss*m1_w + mask2_loss*m2_w + mask3_loss*m3_w + mask4_loss*m4_w
        losses[f'seg_loss'] = seg_loss * self.seg_loss_weights[0]

        cls_loss = self.cls_loss(pred_logit, img_label.squeeze(dim=1))
        losses[f'cls_loss'] = cls_loss * self.seg_loss_weights[1]

        for i, aux_pred_mask in enumerate(aux_pred_masks):
            seg_loss = self.compute_aux_seg_loss(*self.aux_seg_losses, aux_pred_mask, gt_semantic_seg)
            losses[f'aux_seg_loss_{i}'] = seg_loss * self.aux_mask_loss_weights[i]
        
        return losses

    def compute_aux_seg_loss(self, loss_func1, loss_func2, features, gt_mask):
        return loss_func1(features, gt_mask) + loss_func2(features, gt_mask)

    def forward_test(self, imgs, img_metas, **kwargs):
        return self.simple_test(imgs, img_metas, **kwargs)

    def inference(self, img, img_meta, rescale, argmax=False):
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)

        masks, aux_masks, cls = self.base_model(img)
        seg_logit = masks[0]
        aux_mask = aux_masks[0]

        if img_meta[0]['pad_pre_shape'] is not None \
            and img_meta[0]['pad_shape'] is not None \
            and img_meta[0]['pad_shape'] != img_meta[0]['pad_pre_shape']:
            pad_pre_shape = img_meta[0]['pad_pre_shape'][:2]
            seg_logit = seg_logit[:, :, 0:pad_pre_shape[0], 0:pad_pre_shape[1]]
            aux_mask = aux_mask[:, :, 0:pad_pre_shape[0], 0:pad_pre_shape[1]]

        if rescale:
            size = img_meta[0]['ori_shape'][:2]
            seg_logit = F.interpolate(seg_logit, size, mode='bilinear', align_corners=True)
            aux_mask = F.interpolate(aux_mask, size, mode='bilinear', align_corners=True)

        if cls.shape[1] == 1:
            cls = cls.flatten()
            cls_pred = torch.sigmoid(cls)
        else:
            cls_pred = F.softmax(cls, dim=1)
            if argmax:
                cls_pred = cls_pred.argmax(dim=1, keepdim=False)
            else:
                # 2 classes
                cls_pred = cls_pred[:, 1]
        
        if aux_mask.shape[1] == 1:
            aux_seg_pred = torch.sigmoid(aux_mask)
        else:
            aux_seg_pred = F.softmax(aux_mask, dim=1)

        seg_pred = (seg_logit + aux_seg_pred) / 2

        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                seg_pred = seg_pred.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                seg_pred = seg_pred.flip(dims=(2, ))

        return cls_pred, seg_pred

    def simple_test(self, img, img_meta, rescale=True, argmax=False):
        """Simple test with single image."""
        cls_pred, seg_pred = self.inference(img, img_meta, rescale, argmax=argmax)
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


@DETECTORS.register_module()
class PSCCDetectorv3(BaseModule, metaclass=ABCMeta):
    def __init__(self,
                 base_model,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.base_model = build_base_model(base_model)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.seg_loss = (build_loss(train_cfg.seg_loss[0]), build_loss(train_cfg.seg_loss[1]))
        self.seg_loss_weights = train_cfg.seg_loss_weights
        self.mask_loss_weights = train_cfg.get('mask_loss_weights', (1.0, 1.0, 1.0, 1.0))
        self.cls_loss = build_loss(train_cfg.cls_loss)

    def generate_4masks(self, mask):
        h, w = mask.size()[2:]
        mask_tmp = mask.to(torch.float32)
        # print(mask)
        mask2 = torch.nn.functional.interpolate(mask_tmp, size=(h//2, w//2), mode='bilinear', align_corners=True)
        mask2[mask2 > 0.5] = 1
        mask2[mask2 <= 0.5] = 0
        # mask2 = mask2.to(torch.int64)

        mask3 = torch.nn.functional.interpolate(mask_tmp, size=(h//4, w//4), mode='bilinear', align_corners=True)
        mask3[mask3 > 0.5] = 1
        mask3[mask3 <= 0.5] = 0
        # mask3 = mask3.to(torch.int64)

        mask4 = torch.nn.functional.interpolate(mask_tmp, size=(h//8, w//8), mode='bilinear', align_corners=True)
        mask4[mask4 > 0.5] = 1
        mask4[mask4 <= 0.5] = 0
        # mask4 = mask4.to(torch.int64)

        # return mask, mask2, mask3, mask4
        return mask_tmp, mask2, mask3, mask4

    def compute_seg_loss(self, loss_func1, loss_func2, features, gt_mask):
        return loss_func1(features, gt_mask) + loss_func2(features, gt_mask)

    def forward_train(self, 
                    img, 
                    img_metas, 
                    gt_semantic_seg, 
                    img_label=None):
        losses = dict()
        
        (pred_mask1, pred_mask2, pred_mask3, pred_mask4), pred_logit = self.base_model(img)

        mask1, mask2, mask3, mask4 = self.generate_4masks(gt_semantic_seg)

        mask1_loss = self.compute_seg_loss(*self.seg_loss, pred_mask1, mask1)
        mask2_loss = self.compute_seg_loss(*self.seg_loss, pred_mask2, mask2)
        mask3_loss = self.compute_seg_loss(*self.seg_loss, pred_mask3, mask3)
        mask4_loss = self.compute_seg_loss(*self.seg_loss, pred_mask4, mask4)
        m1_w, m2_w, m3_w, m4_w = self.mask_loss_weights
        seg_loss = mask1_loss*m1_w + mask2_loss*m2_w + mask3_loss*m3_w + mask4_loss*m4_w
        losses[f'seg_loss'] = seg_loss * self.seg_loss_weights[0]

        cls_loss = self.cls_loss(pred_logit, img_label.squeeze(dim=1))
        losses[f'cls_loss'] = cls_loss * self.seg_loss_weights[1]
        
        return losses

    def forward_test(self, imgs, img_metas, **kwargs):
        return self.simple_test(imgs, img_metas, **kwargs)

    def inference(self, img, img_meta, rescale, argmax=False):
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)

        masks, cls = self.base_model(img)
        seg_logit = masks[0]

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
            if argmax:
                cls_pred = cls_pred.argmax(dim=1, keepdim=False)
            else:
                # 2 classes
                cls_pred = cls_pred[:, 1]
        
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

        return cls_pred, seg_pred

    def simple_test(self, img, img_meta, rescale=True, argmax=False):
        """Simple test with single image."""
        cls_pred, seg_pred = self.inference(img, img_meta, rescale, argmax=argmax)
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


@DETECTORS.register_module()
class PSCCDetectorv4(PSCCDetectorv3, BaseModule):
    def __init__(self,
                 base_model,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        BaseModule.__init__(self, init_cfg)
        self.base_model = build_base_model(base_model)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.seg_loss = build_loss(train_cfg.seg_loss)
        self.seg_loss_weights = train_cfg.seg_loss_weights
        self.mask_loss_weights = train_cfg.get('mask_loss_weights', (1.0, 1.0, 1.0, 1.0))
        self.cls_loss = build_loss(train_cfg.cls_loss)

    def forward_train(self, 
                    img, 
                    img_metas, 
                    gt_semantic_seg, 
                    img_label=None):
        losses = dict()

        mask1, mask2, mask3, mask4 = self.generate_4masks(gt_semantic_seg)
        
        (pred_mask1, pred_mask2, pred_mask3, pred_mask4), pred_logit = self.base_model(img)

        mask1_loss = self.seg_loss(pred_mask1, mask1)
        mask2_loss = self.seg_loss(pred_mask2, mask2)
        mask3_loss = self.seg_loss(pred_mask3, mask3)
        mask4_loss = self.seg_loss(pred_mask4, mask4)
        m1_w, m2_w, m3_w, m4_w = self.mask_loss_weights
        seg_loss = mask1_loss*m1_w + mask2_loss*m2_w + mask3_loss*m3_w + mask4_loss*m4_w
        losses[f'seg_loss'] = seg_loss * self.seg_loss_weights[0]

        cls_loss = self.cls_loss(pred_logit, img_label.squeeze(dim=1))
        losses[f'cls_loss'] = cls_loss * self.seg_loss_weights[1]
        
        return losses


@DETECTORS.register_module()
class PSCCDetectorv5(PSCCDetector, metaclass=ABCMeta):
    def __init__(self, base_model, train_cfg=None, test_cfg=None, init_cfg=None):
        super().__init__(base_model, train_cfg=train_cfg, test_cfg=test_cfg, init_cfg=init_cfg)

    def generate_4masks(self, top_size, mask):
        h, w = top_size

        mask_tmp = mask.to(torch.float32)
        mask1 = torch.nn.functional.interpolate(mask_tmp, size=(h, w), mode='bilinear', align_corners=True)
        mask1[mask1 > 0.5] = 1
        mask1[mask1 <= 0.5] = 0

        h, w = mask1.size()[2:]

        # print(mask)
        mask2 = torch.nn.functional.interpolate(mask_tmp, size=(h//2, w//2), mode='bilinear', align_corners=True)
        mask2[mask2 > 0.5] = 1
        mask2[mask2 <= 0.5] = 0
        # mask2 = mask2.to(torch.int64)

        mask3 = torch.nn.functional.interpolate(mask_tmp, size=(h//4, w//4), mode='bilinear', align_corners=True)
        mask3[mask3 > 0.5] = 1
        mask3[mask3 <= 0.5] = 0
        # mask3 = mask3.to(torch.int64)

        mask4 = torch.nn.functional.interpolate(mask_tmp, size=(h//8, w//8), mode='bilinear', align_corners=True)
        mask4[mask4 > 0.5] = 1
        mask4[mask4 <= 0.5] = 0
        # mask4 = mask4.to(torch.int64)

        # return mask, mask2, mask3, mask4
        return mask1, mask2, mask3, mask4

    def forward_train(self, 
                    img, 
                    img_metas, 
                    gt_semantic_seg, 
                    img_label=None,
                    qtables=None):
        losses = dict()

        (pred_mask1, pred_mask2, pred_mask3, pred_mask4), pred_logit = self.base_model(img, qtables)

        mask1, mask2, mask3, mask4 = self.generate_4masks(pred_mask1.shape[2:], gt_semantic_seg)
        # median-frequency class weighting
        mask1_balance = torch.ones_like(mask1, dtype=torch.float)
        p_balance_scale = self.p_balance_scale
        n_balance_scale = self.n_balance_scale
        if (mask1 == 1).sum():
            mask1_balance[mask1 == 1] = p_balance_scale / ((mask1 == 1).sum().to(torch.float) / mask1.numel())
            mask1_balance[mask1 == 0] = n_balance_scale / ((mask1 == 0).sum().to(torch.float) / mask1.numel())
        # else:
        #     print('Mask1 balance is not working!')

        mask2_balance = torch.ones_like(mask2, dtype=torch.float)
        if (mask2 == 1).sum():
            mask2_balance[mask2 == 1] = p_balance_scale / ((mask2 == 1).sum().to(torch.float) / mask2.numel())
            mask2_balance[mask2 == 0] = n_balance_scale / ((mask2 == 0).sum().to(torch.float) / mask2.numel())
        # else:
        #     print('Mask2 balance is not working!')

        mask3_balance = torch.ones_like(mask3, dtype=torch.float)
        if (mask3 == 1).sum():
            mask3_balance[mask3 == 1] = p_balance_scale / ((mask3 == 1).sum().to(torch.float) / mask3.numel())
            mask3_balance[mask3 == 0] = n_balance_scale / ((mask3 == 0).sum().to(torch.float) / mask3.numel())
        # else:
        #     print('Mask3 balance is not working!')

        mask4_balance = torch.ones_like(mask4, dtype=torch.float)
        if (mask4 == 1).sum():
            mask4_balance[mask4 == 1] = p_balance_scale / ((mask4 == 1).sum().to(torch.float) / mask4.numel())
            mask4_balance[mask4 == 0] = n_balance_scale / ((mask4 == 0).sum().to(torch.float) / mask4.numel())
        # else:
        #     print('Mask4 balance is not working!')

        # mask1, mask2, mask3, mask4 = mask1.squeeze(dim=1), mask2.squeeze(
        #         dim=1), mask3.squeeze(dim=1), mask4.squeeze(dim=1)

        mask1_loss = torch.mean(self.seg_loss(pred_mask1, mask1) * mask1_balance)
        mask2_loss = torch.mean(self.seg_loss(pred_mask2, mask2) * mask2_balance)
        mask3_loss = torch.mean(self.seg_loss(pred_mask3, mask3) * mask3_balance)
        mask4_loss = torch.mean(self.seg_loss(pred_mask4, mask4) * mask4_balance)
        m1_w, m2_w, m3_w, m4_w = self.mask_loss_weights
        seg_loss = mask1_loss*m1_w + mask2_loss*m2_w + mask3_loss*m3_w + mask4_loss*m4_w
        losses[f'seg_loss'] = seg_loss * self.seg_loss_weights[0]

        cls_loss = self.cls_loss(pred_logit, img_label.squeeze(dim=1))
        losses[f'cls_loss'] = cls_loss * self.seg_loss_weights[1]
        
        return losses

    def inference(self, img, img_meta, qtables, rescale, argmax=False):
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)

        masks, cls = self.base_model(img, qtables)
        seg_logit = masks[0]

        if img_meta[0]['pad_pre_shape'] is not None \
            and img_meta[0]['pad_shape'] is not None \
            and img_meta[0]['pad_shape'] != img_meta[0]['pad_pre_shape']:
            pad_pre_shape = img_meta[0]['pad_pre_shape'][:2]
            seg_logit = seg_logit[:, :, 0:pad_pre_shape[0], 0:pad_pre_shape[1]]

        if rescale:
            size = img_meta[0]['pad_shape']
            seg_logit = F.interpolate(seg_logit, size, mode='bilinear', align_corners=True)

        if cls.shape[1] == 1:
            cls = cls.flatten()
            cls_pred = torch.sigmoid(cls)
        else:
            cls_pred = F.softmax(cls, dim=1)
            if argmax:
                cls_pred = cls_pred.argmax(dim=1, keepdim=False)
            else:
                cls_pred = cls_pred[:, 1]
        
        # if seg_logit.shape[1] == 1:
        #     seg_pred = torch.sigmoid(seg_logit)
        # else:
        #     seg_pred = F.softmax(seg_logit, dim=1)
        #     if argmax:
        #         seg_pred = seg_pred.argmax(dim=1, keepdim=False)
        seg_pred = seg_logit

        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                seg_pred = seg_pred.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                seg_pred = seg_pred.flip(dims=(2, ))

        return cls_pred, seg_pred

    def simple_test(self, img, img_meta, qtables, rescale=True, argmax=False):
        """Simple test with single image."""
        cls_pred, seg_pred = self.inference(img, img_meta, qtables, rescale, argmax=argmax)
        cls_pred = cls_pred.cpu().numpy()
        seg_pred = seg_pred.cpu().numpy()

        return cls_pred, seg_pred
