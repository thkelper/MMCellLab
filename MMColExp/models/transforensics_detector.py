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
class TransForensicsDetector(BaseModule, metaclass=ABCMeta):
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
            self.clf_loss = build_loss(train_cfg.clf_loss)

    def compute_seg_loss(self, loss_func1, loss_func2, features, gt_mask):
        return loss_func1(features, gt_mask) + loss_func2(features, gt_mask)

    def compte_clf_loss(self, cls_pred, gt_label):
        return self.clf_loss(cls_pred, gt_label)

    def forward_train(self, 
                    img, 
                    img_metas, 
                    gt_semantic_seg, 
                    img_label=None):
        losses = dict()

        outs = self.base_model(img)
        seg_loss = 0.0
        for i, (cls_pred, pred_mask) in enumerate(outs):
            seg_loss = self.compute_seg_loss(*self.seg_losses, pred_mask, gt_semantic_seg)
            clf_loss = self.compte_clf_loss(cls_pred, img_label)
            
            losses[f'seg_loss_{i}'] = seg_loss * self.seg_loss_weights[i]
            losses[f'clf_loss_{i}'] = clf_loss * self.seg_loss_weights[i]
        
        return losses

    def forward_test(self, imgs, img_metas, **kwargs):
        return self.simple_test(imgs, img_metas, **kwargs)

    def inference(self, img, img_meta, rescale, argmax=False):
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)

        cls, seg_logit = self.base_model(img)[0]

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


# @DETECTORS.register_module()
# class TransForensicsDetector_v2(BaseModule, metaclass=ABCMeta):
#     def __init__(self,
#                  base_model,
#                  train_cfg=None,
#                  test_cfg=None,
#                  init_cfg=None):
#         super().__init__(init_cfg)
#         self.base_model = build_base_model(base_model)
#         self.train_cfg = train_cfg
#         self.test_cfg = test_cfg

#         self.seg_loss_weights = train_cfg.seg_loss_weights
#         self.ran_loss_weights = train_cfg.ran_loss_weights

#         self.seg0_losses = (build_loss(train_cfg.seg_loss[0]),
#                             build_loss(train_cfg.seg_loss[1]))
#         self.seg1_losses = (build_loss(train_cfg.seg_loss[0]),
#                             build_loss(train_cfg.seg_loss[1]))
#         self.seg2_losses = (build_loss(train_cfg.seg_loss[0]),
#                             build_loss(train_cfg.seg_loss[1]))
#         self.clf_loss = None
#         if 'clf_loss' in train_cfg:
#             self.clf_loss = build_loss(train_cfg.clf_loss)

#     def compute_seg_loss(self, loss_func1, loss_func2, features, gt_mask):
#         return loss_func1(features, gt_mask) + loss_func2(features, gt_mask)

#     def compte_clf_loss(self, cls_pred, gt_label):
#         return self.clf_loss(cls_pred, gt_label)

#     def forward_train(self, 
#                     img, 
#                     img_metas, 
#                     gt_semantic_seg, 
#                     img_label=None):
#         losses = dict()

#         outs = self.base_model(img)
#         for i, (cls_pred, masks) in enumerate(outs):
#             comb_pred, org_pred, rev_pred = masks

#             comb_seg_loss = self.compute_seg_loss(*self.seg0_losses, comb_pred, gt_semantic_seg) * self.ran_loss_weights[0]
#             org_seg_loss = self.compute_seg_loss(*self.seg1_losses, org_pred, gt_semantic_seg) * self.ran_loss_weights[1]
#             rev_seg_loss = self.compute_seg_loss(*self.seg2_losses, rev_pred, gt_semantic_seg) * self.ran_loss_weights[2]

#             clf_loss = self.compte_clf_loss(cls_pred, img_label)
            
#             losses[f'comb_seg_loss_{i}'] = comb_seg_loss * self.seg_loss_weights[i]
#             losses[f'org_seg_loss_{i}'] = org_seg_loss * self.seg_loss_weights[i]
#             losses[f'rev_seg_loss_{i}'] = rev_seg_loss * self.seg_loss_weights[i]
#             losses[f'clf_loss_{i}'] = clf_loss * self.seg_loss_weights[i]
        
#         return losses

#     def forward_test(self, imgs, img_metas, **kwargs):
#         return self.simple_test(imgs, img_metas, **kwargs)

#     def inference(self, img, img_meta, rescale, argmax=False):
#         ori_shape = img_meta[0]['ori_shape']
#         assert all(_['ori_shape'] == ori_shape for _ in img_meta)

#         cls, seg_logit = self.base_model(img)[0]
#         # seg_logit = seg_logit[0]

#         if img_meta[0]['pad_pre_shape'] is not None \
#             and img_meta[0]['pad_shape'] is not None \
#             and img_meta[0]['pad_shape'] != img_meta[0]['pad_pre_shape']:
#             pad_pre_shape = img_meta[0]['pad_pre_shape'][:2]
#             seg_logit = seg_logit[:, :, 0:pad_pre_shape[0], 0:pad_pre_shape[1]]

#         if rescale:
#             size = img_meta[0]['ori_shape'][:2]
#             seg_logit = F.interpolate(seg_logit, size, mode='bilinear', align_corners=True)

#         if cls.shape[1] == 1:
#             cls = cls.flatten()
#             cls_pred = torch.sigmoid(cls)
#         else:
#             cls_pred = F.softmax(cls, dim=1)
#             if argmax:
#                 cls_pred = cls_pred.argmax(dim=1, keepdim=False)
        
#         if seg_logit.shape[1] == 1:
#             seg_pred = torch.sigmoid(seg_logit)
#         else:
#             seg_pred = F.softmax(seg_logit, dim=1)
#             if argmax:
#                 seg_pred = seg_pred.argmax(dim=1, keepdim=False)

#         flip = img_meta[0]['flip']
#         if flip:
#             flip_direction = img_meta[0]['flip_direction']
#             assert flip_direction in ['horizontal', 'vertical']
#             if flip_direction == 'horizontal':
#                 seg_pred = seg_pred.flip(dims=(3, ))
#             elif flip_direction == 'vertical':
#                 seg_pred = seg_pred.flip(dims=(2, ))

#         return cls_pred, seg_pred

#     def simple_test(self, img, img_meta, rescale=True, argmax=False):
#         """Simple test with single image."""
#         cls_pred, seg_pred = self.inference(img, img_meta, rescale, argmax=argmax)
#         cls_pred = cls_pred.cpu().numpy()
#         seg_pred = seg_pred.cpu().numpy()

#         return cls_pred, seg_pred

#     # @auto_fp16(apply_to=('img',))
#     @force_fp32(apply_to=('img',))
#     def forward(self, img, img_metas, return_loss=True, **kwargs):
#         if return_loss:
#             return self.forward_train(img, img_metas, **kwargs)
#         else:
#             return self.forward_test(img, img_metas, **kwargs)

#     def train_step(self, data_batch, *args, **kwargs):
#         losses = self(**data_batch)
#         loss, log_vars = self._parse_losses(losses)

#         outputs = dict(
#             loss=loss,
#             log_vars=log_vars,
#             num_samples=len(data_batch['img_metas']))

#         return outputs

#     @staticmethod
#     def _parse_losses(losses):
#         log_vars = OrderedDict()
#         for loss_name, loss_value in losses.items():
#             if isinstance(loss_value, torch.Tensor):
#                 log_vars[loss_name] = loss_value.mean()
#             elif isinstance(loss_value, list):
#                 log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
#             else:
#                 raise TypeError(
#                     f'{loss_name} is not a tensor or list of tensors')

#         loss = sum(_value for _key, _value in log_vars.items()
#                    if 'loss' in _key)

#         # If the loss_vars has different length, raise assertion error
#         # to prevent GPUs from infinite waiting.
#         if dist.is_available() and dist.is_initialized():
#             log_var_length = torch.tensor(len(log_vars), device=loss.device)
#             dist.all_reduce(log_var_length)
#             message = (f'rank {dist.get_rank()}' +
#                        f' len(log_vars): {len(log_vars)}' + ' keys: ' +
#                        ','.join(log_vars.keys()) + '\n')
#             assert log_var_length == len(log_vars) * dist.get_world_size(), \
#                 'loss log variables are different across GPUs!\n' + message

#         log_vars['loss'] = loss
#         for loss_name, loss_value in log_vars.items():
#             # reduce loss when distributed training
#             if dist.is_available() and dist.is_initialized():
#                 loss_value = loss_value.data.clone()
#                 dist.all_reduce(loss_value.div_(dist.get_world_size()))
#             log_vars[loss_name] = loss_value.item()

#         return loss, log_vars


# @DETECTORS.register_module()
# class TransForensicsDetectorv3(TransForensicsDetector, BaseModule):
#     def __init__(self,
#                  base_model,
#                  train_cfg=None,
#                  test_cfg=None,
#                  init_cfg=None):
#         BaseModule.__init__(self, init_cfg)
#         self.base_model = build_base_model(base_model)
#         self.train_cfg = train_cfg
#         self.test_cfg = test_cfg

#         self.seg_loss_weights = train_cfg.seg_loss_weights

#         self.seg_loss = build_loss(train_cfg.seg_loss)
#         self.clf_loss = None
#         if 'clf_loss' in train_cfg:
#             self.clf_loss = build_loss(train_cfg.clf_loss)

#     # def compute_seg_loss(self, loss_func1, loss_func2, features, gt_mask):
#     #     return loss_func1(features, gt_mask) + loss_func2(features, gt_mask)

#     def compte_clf_loss(self, cls_pred, gt_label):
#         return self.clf_loss(cls_pred, gt_label)

#     def forward_train(self, 
#                     img, 
#                     img_metas, 
#                     gt_semantic_seg, 
#                     img_label=None):
#         losses = dict()

#         outs = self.base_model(img)
#         seg_loss = 0.0
#         for i, (cls_pred, pred_mask) in enumerate(outs):
#             seg_loss = self.seg_loss(pred_mask, gt_semantic_seg)
#             clf_loss = self.compte_clf_loss(cls_pred, img_label)
            
#             losses[f'seg_loss_{i}'] = seg_loss * self.seg_loss_weights[i]
#             losses[f'clf_loss_{i}'] = clf_loss * self.seg_loss_weights[i]
        
#         return losses


# @DETECTORS.register_module()
# class TransForensicsDetectorv4(TransForensicsDetector):
#     """
#     Add inputs of DCT coefficients in CAT-Network.
#     """
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

#     def forward(self, img, dct_coef, img_metas, return_loss=True, **kwargs):
#         img = torch.cat([img, dct_coef], dim=1)
#         return super().forward(img, img_metas, return_loss=return_loss, **kwargs)


@DETECTORS.register_module()
class TransForensicsDetectorv5(TransForensicsDetector):
    """
    Add inputs of DCT volume in CAT-Network.
    """
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)

    @force_fp32(apply_to=('img',))
    def forward(self, img, dct_vol, qtables, img_metas, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img,  dct_vol, qtables, img_metas, **kwargs)
        else:
            return self.forward_test(img,  dct_vol, qtables, img_metas, **kwargs)

    def forward_train(self, 
                    img, 
                    dct_vol, 
                    qtables,
                    img_metas, 
                    gt_semantic_seg, 
                    img_label=None):
        losses = dict()

        outs = self.base_model(img, dct_vol, qtables)
        seg_loss = 0.0
        for i, (cls_pred, pred_mask) in enumerate(outs):
            seg_loss = self.compute_seg_loss(*self.seg_losses, pred_mask, gt_semantic_seg)
            clf_loss = self.compte_clf_loss(cls_pred, img_label)
            
            losses[f'seg_loss_{i}'] = seg_loss * self.seg_loss_weights[i]
            losses[f'clf_loss_{i}'] = clf_loss * self.seg_loss_weights[i]
        
        return losses

    def forward_test(self, imgs, dct_vol, qtables, img_metas, **kwargs):
        return self.simple_test(imgs, dct_vol, qtables, img_metas, **kwargs)

    def inference(self, img, dct_vol, qtables, img_meta, rescale, argmax=False):
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)

        cls, seg_logit = self.base_model(img, dct_vol, qtables)[0]

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

    def simple_test(self, img, dct_vol, qtables, img_meta, rescale=True, argmax=False):
        """Simple test with single image."""
        cls_pred, seg_pred = self.inference(img, dct_vol, qtables, img_meta, rescale, argmax=argmax)
        cls_pred = cls_pred.cpu().numpy()
        seg_pred = seg_pred.cpu().numpy()

        return cls_pred, seg_pred


@DETECTORS.register_module()
class TransForensicsDetectorv6(TransForensicsDetector):
    """
    Add inputs of DCT volume in CAT-Network.
    Ensemble cls_score to seg_score.
    """
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)

    @force_fp32(apply_to=('img',))
    def forward(self, img, dct_vol, qtables, img_metas, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img,  dct_vol, qtables, img_metas, **kwargs)
        else:
            return self.forward_test(img,  dct_vol, qtables, img_metas, **kwargs)

    def compute_seg_loss(self, loss_func1, loss_func2, features, gt_mask, pos):
        avg = pos.sum() + 1e-6
        loss1 = loss_func1(features, gt_mask) 
        loss1 = (loss1 * pos).sum() / avg
        loss2 = loss_func2(features, gt_mask)
        loss2 = (loss2.flatten(1).mean(dim=1)*pos).sum() / avg

        return loss1 + loss2

    def forward_train(self, 
                    img, 
                    dct_vol, 
                    qtables,
                    img_metas, 
                    gt_semantic_seg, 
                    img_label=None):
        losses = dict()

        outs = self.base_model(img, dct_vol, qtables)
        seg_loss = 0.0
        for i, (cls_pred, pred_mask) in enumerate(outs):
            # only positive samples compute seg loss
            pos = (img_label > 0).to(torch.float32)
            seg_loss = self.compute_seg_loss(*self.seg_losses, pred_mask,gt_semantic_seg, pos)
            losses[f'seg_loss_{i}'] = seg_loss * self.seg_loss_weights[i]
            
            clf_loss = self.compte_clf_loss(cls_pred, img_label)
            losses[f'clf_loss_{i}'] = clf_loss * self.seg_loss_weights[i]
        
        return losses

    def forward_test(self, imgs, dct_vol, qtables, img_metas, **kwargs):
        return self.simple_test(imgs, dct_vol, qtables, img_metas, **kwargs)

    def inference(self, img, dct_vol, qtables, img_meta, rescale, argmax=False):
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)

        cls, seg_logit = self.base_model(img, dct_vol, qtables)[0]

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
            # if argmax:
            #     cls_pred = cls_pred.argmax(dim=1, keepdim=False)
            cls_pred = cls_pred[:, 1]
        
        if seg_logit.shape[1] == 1:
            seg_pred = torch.sigmoid(seg_logit)
        else:
            seg_pred = F.softmax(seg_logit, dim=1)
            seg_pred = cls_pred[:, 1, ...]
            # if argmax:
            #     seg_pred = seg_pred.argmax(dim=1, keepdim=False)

        # conditional prob
        seg_pred = seg_pred * cls_pred[..., None, None]

        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                seg_pred = seg_pred.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                seg_pred = seg_pred.flip(dims=(2, ))

        return cls_pred, seg_pred

    def simple_test(self, img, dct_vol, qtables, img_meta, rescale=True, argmax=False):
        """Simple test with single image."""
        cls_pred, seg_pred = self.inference(img, dct_vol, qtables, img_meta, rescale, argmax=argmax)
        cls_pred = cls_pred.cpu().numpy()
        seg_pred = seg_pred.cpu().numpy()

        return cls_pred, seg_pred

@DETECTORS.register_module()
class TransForensicsDetectorv7(TransForensicsDetector):
    """
    Ensemble cls_score to seg_score.
    """
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)

    @force_fp32(apply_to=('img',))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def compute_seg_loss(self, loss_func1, loss_func2, features, gt_mask, pos):
        avg = pos.sum() + 1e-6
        loss1 = loss_func1(features, gt_mask) 
        loss1 = (loss1 * pos).sum() / avg
        loss2 = loss_func2(features, gt_mask)
        loss2 = (loss2.flatten(1).mean(dim=1)*pos).sum() / avg

        return loss1 + loss2

    def forward_train(self, 
                    img, 
                    img_metas, 
                    gt_semantic_seg, 
                    img_label=None):
        losses = dict()

        outs = self.base_model(img)
        seg_loss = 0.0
        for i, (cls_pred, pred_mask) in enumerate(outs):
            # only positive samples compute seg loss
            pos = (img_label > 0).to(torch.float32)
            seg_loss = self.compute_seg_loss(*self.seg_losses, pred_mask,gt_semantic_seg, pos)
            losses[f'seg_loss_{i}'] = seg_loss * self.seg_loss_weights[i]
            
            clf_loss = self.compte_clf_loss(cls_pred, img_label)
            losses[f'clf_loss_{i}'] = clf_loss * self.seg_loss_weights[i]
        
        return losses

    def forward_test(self, imgs, img_metas, **kwargs):
        return self.simple_test(imgs, img_metas, **kwargs)

    def inference(self, img, img_meta, rescale, argmax=False):
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)

        cls, seg_logit = self.base_model(img)[0]

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
            # if argmax:
            #     cls_pred = cls_pred.argmax(dim=1, keepdim=False)
            cls_pred = cls_pred[:, 1]
        
        if seg_logit.shape[1] == 1:
            seg_pred = torch.sigmoid(seg_logit)
        else:
            seg_pred = F.softmax(seg_logit, dim=1)
            seg_pred = cls_pred[:, 1, ...]
            # if argmax:
            #     seg_pred = seg_pred.argmax(dim=1, keepdim=False)

        # conditional prob
        seg_pred = seg_pred * cls_pred[..., None, None]

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