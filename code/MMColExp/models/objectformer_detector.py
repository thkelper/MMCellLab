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
class ObjectFormerDetector(torch.nn.Module):
    def __init__(self,
                 base_model,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        self.base_model = build_base_model(base_model)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.seg_loss = build_loss(train_cfg.seg_loss)
        self.clf_loss = None
        if 'clf_loss' in train_cfg:
            self.clf_loss = build_loss(train_cfg.clf_loss)

    def compute_seg_loss(self, loss_func, features, gt_mask):
        return loss_func(features, gt_mask)

    def compte_clf_loss(self, cls_pred, gt_label):
        return self.clf_loss(cls_pred, gt_label)

    def forward_train(self, 
                    img, 
                    high_freq_img,
                    img_metas, 
                    gt_semantic_seg, 
                    img_label=None,
                    edge_gt_semantic_seg=None):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()

        cls_pred, pred_mask = self.base_model(img, high_freq_img)
        # seg_loss = self.compute_seg_loss(self.seg_loss, pred_mask, gt_semantic_seg)
        seg_loss = self.seg_loss(pred_mask, gt_semantic_seg)
        losses['seg_loss'] = seg_loss

        if self.clf_loss:
            clf_loss = self.compte_clf_loss(cls_pred, img_label)
            losses['clf_loss'] = clf_loss

        return losses

    def forward_test(self, imgs, high_freq_img, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        return self.simple_test(imgs, high_freq_img, img_metas, **kwargs)

    def inference(self, img, high_freq_img, img_meta, rescale, argmax=True):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)

        cls, seg_logit = self.base_model(img, high_freq_img)

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

    def simple_test(self, img, high_freq_img, img_meta, rescale=True, argmax=True):
        """Simple test with single image."""
        cls_pred, seg_pred = self.inference(img, high_freq_img, img_meta, rescale, argmax=argmax)
        cls_pred = cls_pred.cpu().numpy()
        seg_pred = seg_pred.cpu().numpy()

        return cls_pred, seg_pred

    # @auto_fp16(apply_to=('img',))
    @force_fp32(apply_to=('img',))
    def forward(self, img, high_freq_img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, high_freq_img, img_metas, **kwargs)
        else:
            return self.forward_test(img, high_freq_img, img_metas, **kwargs)

    def train_step(self, data_batch, *args, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        losses = self(**data_batch)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data_batch['img_metas']))

        return outputs

    @staticmethod
    def _parse_losses(losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
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
