import torch
import torch.nn as nn
import torch.nn.functional as F
# from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss
from .builder import LOSSES
from .utils import get_class_weight, weighted_loss, weight_reduce_loss


LOSSES.register_module(nn.BCELoss)

@LOSSES.register_module()
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input.float(), target.float())
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


@weighted_loss
def dice_loss(pred,
              target,
              valid_mask,
              smooth=1,
              exponent=2,
              class_weight=None,
              ignore_index=255):
    assert pred.shape[0] == target.shape[0]
    total_loss = 0
    num_classes = pred.shape[1]
    for i in range(num_classes):
        if i != ignore_index:
            dice_loss = binary_dice_loss(
                pred[:, i],
                target[..., i],
                valid_mask=valid_mask,
                smooth=smooth,
                exponent=exponent)
            if class_weight is not None:
                dice_loss *= class_weight[i]
            total_loss += dice_loss
    return total_loss / num_classes


@weighted_loss
def binary_dice_loss(pred, target, valid_mask=None, smooth=1, exponent=2, **kwards):
    assert pred.shape[0] == target.shape[0]
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)

    if valid_mask is not None:
        valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)
        num = torch.sum(torch.mul(pred, target) * valid_mask, dim=1) * 2 + smooth
    else:
        num = torch.sum(torch.mul(pred, target), dim=1) * 2 + smooth
    den = torch.sum(pred.pow(exponent) + target.pow(exponent), dim=1) + smooth

    return 1 - num / den


@LOSSES.register_module()
class DiceLoss(nn.Module):
    """DiceLoss.

    This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
    Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.

    Args:
        smooth (float): A float number to smooth loss, and avoid NaN error.
            Default: 1
        exponent (float): An float number to calculate denominator
            value: \\sum{x^exponent} + \\sum{y^exponent}. Default: 2.
        reduction (str, optional): The method used to reduce the loss. Options
            are "none", "mean" and "sum". This parameter only works when
            per_image is True. Default: 'mean'.
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Default to 1.0.
        ignore_index (int | None): The label index to be ignored. Default: 255.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_dice'.
    """

    def __init__(self,
                 smooth=1,
                 exponent=2,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 ignore_index=255,
                 loss_name='loss_dice',
                 **kwards):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.reduction = reduction
        self.class_weight = get_class_weight(class_weight)
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                avg_factor=None,
                reduction_override=None,
                **kwards):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        one_hot_target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1),
            num_classes=num_classes)
        valid_mask = (target != self.ignore_index).long()

        loss = self.loss_weight * dice_loss(
            pred,
            one_hot_target,
            valid_mask=valid_mask,
            reduction=reduction,
            avg_factor=avg_factor,
            smooth=self.smooth,
            exponent=self.exponent,
            class_weight=class_weight,
            ignore_index=self.ignore_index)
        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name


@LOSSES.register_module()
class BinaryDiceLoss(nn.Module):
    def __init__(self,
                 smooth=1,
                 exponent=2,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 loss_name='binary_loss_dice',
                 ignore_index=255,
                 **kwards):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.reduction = reduction
        self.class_weight = get_class_weight(class_weight)
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        self.ignore_index = ignore_index

    def forward(self,
                pred,
                target,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        pred = torch.sigmoid(pred)

        loss = self.loss_weight * binary_dice_loss(
            pred,
            target,
            valid_mask=target,
            reduction=reduction,
            avg_factor=avg_factor,
            smooth=self.smooth,
            exponent=self.exponent,
            class_weight=class_weight,
            ignore_index=self.ignore_index)
        return loss

    @property
    def loss_name(self):
        return self._loss_name


@weighted_loss
def cross_entropy(pred,
                  label,
                  weight=None,
                  class_weight=None,
                  reduction='mean',
                  avg_factor=None,
                  ignore_index=-100):
    """The wrapper function for :func:`F.cross_entropy`"""
    # class_weight is a manual rescaling weight given to each class.
    # If given, has to be a Tensor of size C element-wise losses
    loss = F.cross_entropy(
        pred,
        label,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def _expand_onehot_labels(labels, label_weights, target_shape, ignore_index):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_zeros(target_shape)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(valid_mask, as_tuple=True)

    if inds[0].numel() > 0:
        if labels.dim() == 3:
            bin_labels[inds[0], labels[valid_mask], inds[1], inds[2]] = 1
        else:
            bin_labels[inds[0], labels[valid_mask]] = 1

    valid_mask = valid_mask.unsqueeze(1).expand(target_shape).float()
    if label_weights is None:
        bin_label_weights = valid_mask
    else:
        bin_label_weights = label_weights.unsqueeze(1).expand(target_shape)
        bin_label_weights *= valid_mask

    return bin_labels, bin_label_weights


@weighted_loss
def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None,
                         ignore_index=255):
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored. Default: 255

    Returns:
        torch.Tensor: The calculated loss
    """
    if pred.dim() != label.dim():
        assert (pred.dim() == 2 and label.dim() == 1) or (
                pred.dim() == 4 and label.dim() == 3), \
            'Only pred shape [N, C], label shape [N] or pred shape [N, C, ' \
            'H, W], label shape [N, H, W] are supported'
        label, weight = _expand_onehot_labels(label, weight, pred.shape,
                                              ignore_index)

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), pos_weight=class_weight, reduction='none')
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def mask_cross_entropy(pred,
                       target,
                       label,
                       reduction='mean',
                       avg_factor=None,
                       class_weight=None,
                       ignore_index=None):
    """Calculate the CrossEntropy loss for masks.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        label (torch.Tensor): ``label`` indicates the class label of the mask'
            corresponding object. This will be used to select the mask in the
            of the class which the object belongs to when the mask prediction
            if not class-agnostic.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (None): Placeholder, to be consistent with other loss.
            Default: None.

    Returns:
        torch.Tensor: The calculated loss
    """
    assert ignore_index is None, 'BCE loss does not support ignore_index'
    # TODO: handle these two reserved arguments
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, weight=class_weight, reduction='mean')[None]


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    """CrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 loss_name='loss_ce'):
        super(CrossEntropyLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = get_class_weight(class_weight)

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy
        self._loss_name = loss_name

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name


def py_sigmoid_focal_loss(pred,
                          target,
                          one_hot_target=None,
                          weight=None,
                          gamma=2.0,
                          alpha=0.5,
                          class_weight=None,
                          valid_mask=None,
                          reduction='mean',
                          avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction with
            shape (N, C)
        one_hot_target (None): Placeholder. It should be None.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float | list[float], optional): A balanced form for Focal Loss.
            Defaults to 0.5.
        class_weight (list[float], optional): Weight of each class.
            Defaults to None.
        valid_mask (torch.Tensor, optional): A mask uses 1 to mark the valid
            samples and uses 0 to mark the ignored samples. Default: None.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    if isinstance(alpha, list):
        alpha = pred.new_tensor(alpha)
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    one_minus_pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * one_minus_pt.pow(gamma)

    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    final_weight = torch.ones(1, pred.size(1)).type_as(loss)
    if weight is not None:
        if weight.shape != loss.shape and weight.size(0) == loss.size(0):
            # For most cases, weight is of shape (N, ),
            # which means it does not have the second axis num_class
            weight = weight.view(-1, 1)
        assert weight.dim() == loss.dim()
        final_weight = final_weight * weight
    if class_weight is not None:
        final_weight = final_weight * pred.new_tensor(class_weight)
    if valid_mask is not None:
        final_weight = final_weight * valid_mask
    loss = weight_reduce_loss(loss, final_weight, reduction, avg_factor)
    return loss


# def sigmoid_focal_loss(pred,
#                        target,
#                        one_hot_target,
#                        weight=None,
#                        gamma=2.0,
#                        alpha=0.5,
#                        class_weight=None,
#                        valid_mask=None,
#                        reduction='mean',
#                        avg_factor=None):
#     r"""A warpper of cuda version `Focal Loss
#     <https://arxiv.org/abs/1708.02002>`_.
#     Args:
#         pred (torch.Tensor): The prediction with shape (N, C), C is the number
#             of classes.
#         target (torch.Tensor): The learning label of the prediction. It's shape
#             should be (N, )
#         one_hot_target (torch.Tensor): The learning label with shape (N, C)
#         weight (torch.Tensor, optional): Sample-wise loss weight.
#         gamma (float, optional): The gamma for calculating the modulating
#             factor. Defaults to 2.0.
#         alpha (float | list[float], optional): A balanced form for Focal Loss.
#             Defaults to 0.5.
#         class_weight (list[float], optional): Weight of each class.
#             Defaults to None.
#         valid_mask (torch.Tensor, optional): A mask uses 1 to mark the valid
#             samples and uses 0 to mark the ignored samples. Default: None.
#         reduction (str, optional): The method used to reduce the loss into
#             a scalar. Defaults to 'mean'. Options are "none", "mean" and "sum".
#         avg_factor (int, optional): Average factor that is used to average
#             the loss. Defaults to None.
#     """
#     # Function.apply does not accept keyword arguments, so the decorator
#     # "weighted_loss" is not applicable
#     final_weight = torch.ones(1, pred.size(1)).type_as(pred)
#     if isinstance(alpha, list):
#         # _sigmoid_focal_loss doesn't accept alpha of list type. Therefore, if
#         # a list is given, we set the input alpha as 0.5. This means setting
#         # equal weight for foreground class and background class. By
#         # multiplying the loss by 2, the effect of setting alpha as 0.5 is
#         # undone. The alpha of type list is used to regulate the loss in the
#         # post-processing process.
#         loss = _sigmoid_focal_loss(pred.contiguous(), target.contiguous(),
#                                    gamma, 0.5, None, 'none') * 2
#         alpha = pred.new_tensor(alpha)
#         final_weight = final_weight * (
#             alpha * one_hot_target + (1 - alpha) * (1 - one_hot_target))
#     else:
#         loss = _sigmoid_focal_loss(pred.contiguous(), target.contiguous(),
#                                    gamma, alpha, None, 'none')
#     if weight is not None:
#         if weight.shape != loss.shape and weight.size(0) == loss.size(0):
#             # For most cases, weight is of shape (N, ),
#             # which means it does not have the second axis num_class
#             weight = weight.view(-1, 1)
#         assert weight.dim() == loss.dim()
#         final_weight = final_weight * weight
#     if class_weight is not None:
#         final_weight = final_weight * pred.new_tensor(class_weight)
#     if valid_mask is not None:
#         final_weight = final_weight * valid_mask
#     loss = weight_reduce_loss(loss, final_weight, reduction, avg_factor)
#     return loss


@LOSSES.register_module()
class FocalLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.5,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 loss_name='loss_focal'):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_
        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float | list[float], optional): A balanced form for Focal
                Loss. Defaults to 0.5. When a list is provided, the length
                of the list should be equal to the number of classes.
                Please be careful that this parameter is not the
                class-wise weight but the weight of a binary classification
                problem. This binary classification problem regards the
                pixels which belong to one class as the foreground
                and the other pixels as the background, each element in
                the list is the weight of the corresponding foreground class.
                The value of alpha or each element of alpha should be a float
                in the interval [0, 1]. If you want to specify the class-wise
                weight, please use `class_weight` parameter.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            loss_name (str, optional): Name of the loss item. If you want this
                loss item to be included into the backward graph, `loss_` must
                be the prefix of the name. Defaults to 'loss_focal'.
        """
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, \
            'AssertionError: Only sigmoid focal loss supported now.'
        assert reduction in ('none', 'mean', 'sum'), \
            "AssertionError: reduction should be 'none', 'mean' or " \
            "'sum'"
        assert isinstance(alpha, (float, list)), \
            'AssertionError: alpha should be of type float'
        assert isinstance(gamma, float), \
            'AssertionError: gamma should be of type float'
        assert isinstance(loss_weight, float), \
            'AssertionError: loss_weight should be of type float'
        assert isinstance(loss_name, str), \
            'AssertionError: loss_name should be of type str'
        assert isinstance(class_weight, list) or class_weight is None, \
            'AssertionError: class_weight must be None or of type list'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.class_weight = class_weight
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction with shape
                (N, C) where C = number of classes, or
                (N, C, d_1, d_2, ..., d_K) with K≥1 in the
                case of K-dimensional loss.
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤C−1,
                or (N, d_1, d_2, ..., d_K) with K≥1 in the case of
                K-dimensional loss. If containing class probabilities,
                same shape as the input.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to
                average the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used
                to override the original reduction method of the loss.
                Options are "none", "mean" and "sum".
            ignore_index (int, optional): The label index to be ignored.
                Default: 255
        Returns:
            torch.Tensor: The calculated loss
        """
        assert isinstance(ignore_index, int), \
            'ignore_index must be of type int'
        assert reduction_override in (None, 'none', 'mean', 'sum'), \
            "AssertionError: reduction should be 'none', 'mean' or " \
            "'sum'"
        assert pred.shape == target.shape or \
               (pred.size(0) == target.size(0) and
                pred.shape[2:] == target.shape[1:]), \
               "The shape of pred doesn't match the shape of target"

        original_shape = pred.shape

        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()

        if original_shape == target.shape:
            # target with shape [B, C, d_1, d_2, ...]
            # transform it's shape into [N, C]
            # [B, C, d_1, d_2, ...] -> [C, B, d_1, d_2, ..., d_k]
            target = target.transpose(0, 1)
            # [C, B, d_1, d_2, ..., d_k] -> [C, N]
            target = target.reshape(target.size(0), -1)
            # [C, N] -> [N, C]
            target = target.transpose(0, 1).contiguous()
        else:
            # target with shape [B, d_1, d_2, ...]
            # transform it's shape into [N, ]
            target = target.view(-1).contiguous()
            valid_mask = (target != ignore_index).view(-1, 1)
            # avoid raising error when using F.one_hot()
            target = torch.where(target == ignore_index, target.new_tensor(0),
                                 target)

        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            num_classes = pred.size(1)
            # if torch.cuda.is_available() and pred.is_cuda:
            #     if target.dim() == 1:
            #         one_hot_target = F.one_hot(target, num_classes=num_classes)
            #     else:
            #         one_hot_target = target
            #         target = target.argmax(dim=1)
            #         valid_mask = (target != ignore_index).view(-1, 1)
            #     calculate_loss_func = sigmoid_focal_loss
            # else:
            one_hot_target = None
            if target.dim() == 1:
                target = F.one_hot(target, num_classes=num_classes)
            else:
                valid_mask = (target.argmax(dim=1) != ignore_index).view(
                    -1, 1)
            calculate_loss_func = py_sigmoid_focal_loss

            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target,
                one_hot_target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                class_weight=self.class_weight,
                valid_mask=valid_mask,
                reduction=reduction,
                avg_factor=avg_factor)

            if reduction == 'none':
                # [N, C] -> [C, N]
                loss_cls = loss_cls.transpose(0, 1)
                # [C, N] -> [C, B, d1, d2, ...]
                # original_shape: [B, C, d1, d2, ...]
                loss_cls = loss_cls.reshape(original_shape[1],
                                            original_shape[0],
                                            *original_shape[2:])
                # [C, B, d1, d2, ...] -> [B, C, d1, d2, ...]
                loss_cls = loss_cls.transpose(0, 1).contiguous()
        else:
            raise NotImplementedError
        return loss_cls

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name


@LOSSES.register_module()
class CrossEntropyLossLabelSmooth(nn.Module):
    def __init__(self, num_cls=2, epsilon=0.1, loss_weight=1.0):
        """ 实现了带label smooth功能的交叉熵损失函数

        Args:
            num_cls (int): 类别数量；
            epsilon (float): label smooth超参，用于控制label平滑化的程度；

        Notes:
            文献:
                Rethinking the Inception Architecture for Computer Vision(https://arxiv.org/abs/1512.00567)
        """
        super(CrossEntropyLossLabelSmooth, self).__init__()
        self.num_cls = num_cls
        self.epsilon = epsilon
        self.loss_weight = loss_weight

    def forward(self, outputs, targets, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        N = targets.size(0)
        if len(targets.shape) == 1:
             targets = torch.unsqueeze(targets, dim=1)
        smoothed_labels = torch.full(size=(N, self.num_cls), 
                                    fill_value=epsilon/(self.num_cls-1)).to(targets.device)
        smoothed_labels.scatter_(dim=1, index=targets, value=1.0-epsilon)
        log_prob = nn.functional.log_softmax(outputs, dim=1)
        loss = - torch.sum(log_prob * smoothed_labels) / N
        loss = loss * self.loss_weight
        return loss
