from collections import OrderedDict, defaultdict
from itertools import product
from operator import add
import os.path as osp
import warnings
import cv2
import mmcv
import numpy as np
from sklearn import metrics
import torch
import torch.distributed as dist
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.runner import DistEvalHook as _DistEvalHook
from mmcv.runner import EvalHook as _EvalHook
from mmcv.utils import print_log
from mmcv.runner import get_dist_info
from collections import OrderedDict
import torch
from prettytable import PrettyTable
from .logger import get_root_logger
from .test import multi_gpu_test


def f_score(precision, recall, beta=1):
    """calculate the f-score value.

    Args:
        precision (float | torch.Tensor): The precision value.
        recall (float | torch.Tensor): The recall value.
        beta (int): Determines the weight of recall in the combined score.
            Default: False.

    Returns:
        [torch.tensor]: The f-score value.
    """
    score = (1 + beta**2) * (precision * recall) / (
        (beta**2 * precision) + recall)
    return score


def calculate_img_score(pd, gt):
    seg_inv, gt_inv = np.logical_not(pd), np.logical_not(gt)
    true_pos = float(np.logical_and(pd, gt).sum())
    false_pos = np.logical_and(pd, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, gt).sum()
    true_neg = float(np.logical_and(seg_inv, gt_inv).sum())
    acc = (true_pos + true_neg) / (true_pos + true_neg + false_neg + false_pos + 1e-6)
    # sen = true_pos / (true_pos + false_neg + 1e-6)
    # spe = true_neg / (true_neg + false_pos + 1e-6)
    # f1 = 2 * sen * spe / (sen + spe)
    # return f1, sen, spe, acc, true_pos, true_neg, false_pos, false_neg
    precision = true_pos / (true_pos + false_pos + 1e-6)
    recall = true_pos / (true_pos + false_neg + 1e-6)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall, acc, true_pos, true_neg, false_pos, false_neg


def calculate_auc(pred, gt):
    auc = metrics.roc_auc_score(gt, pred)
    return auc


def calculate_pixel_f1(pd, gt):
    if np.max(pd) == np.max(gt) and np.max(pd) == 0:
        f1, iou = 1.0, 1.0
        return f1, 0.0, 0.0
    seg_inv, gt_inv = np.logical_not(pd), np.logical_not(gt)
    true_pos = float(np.logical_and(pd, gt).sum())
    false_pos = np.logical_and(pd, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, gt).sum()
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    precision = true_pos / (true_pos + false_pos + 1e-6)
    recall = true_pos / (true_pos + false_neg + 1e-6)

    return f1, precision, recall


def intersect_and_union(pred_label,
                        label,
                        num_classes,
                        ignore_index=None,
                        label_map=dict(),
                        reduce_zero_label=False):
    """Calculate intersection and Union.

    Args:
        pred_label (ndarray | str): Prediction segmentation map
            or predict result filename.
        label (ndarray | str): Ground truth segmentation map
            or label filename.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. The parameter will
            work only when label is str. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. The parameter will
            work only when label is str. Default: False.

     Returns:
         torch.Tensor: The intersection of prediction and ground truth
            histogram on all classes.
         torch.Tensor: The union of prediction and ground truth histogram on
            all classes.
         torch.Tensor: The prediction histogram on all classes.
         torch.Tensor: The ground truth histogram on all classes.
    """

    if isinstance(pred_label, str):
        pred_label = torch.from_numpy(np.load(pred_label))
    elif isinstance(pred_label, np.ndarray):
        pred_label = torch.from_numpy((pred_label))

    if isinstance(label, str):
        label = torch.from_numpy(
            mmcv.imread(label, flag='unchanged', backend='pillow'))
    elif isinstance(label, np.ndarray):
        label = torch.from_numpy(label)

    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    if ignore_index is not None:
        mask = (label != ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect = torch.histc(
        intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_pred_label = torch.histc(
        pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_label = torch.histc(
        label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(results,
                              gt_seg_maps,
                              num_classes,
                              ignore_index,
                              label_map=dict(),
                              reduce_zero_label=False):
    """Calculate Total Intersection and Union.

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str] | Iterables): list of ground
            truth segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """
    total_area_intersect = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_union = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_pred_label = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_label = torch.zeros((num_classes, ), dtype=torch.float64)
    for result, gt_seg_map in zip(results, gt_seg_maps):
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(
                result, gt_seg_map, num_classes, ignore_index,
                label_map, reduce_zero_label)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label


def mean_iou(results,
             gt_seg_maps,
             num_classes,
             ignore_index,
             nan_to_num=None,
             label_map=dict(),
             reduce_zero_label=False):
    """Calculate Mean Intersection and Union (mIoU)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.

     Returns:
        dict[str, float | ndarray]:
            <aAcc> float: Overall accuracy on all images.
            <Acc> ndarray: Per category accuracy, shape (num_classes, ).
            <IoU> ndarray: Per category IoU, shape (num_classes, ).
    """
    iou_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mIoU'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return iou_result


def mean_dice(results,
              gt_seg_maps,
              num_classes,
              ignore_index,
              nan_to_num=None,
              label_map=dict(),
              reduce_zero_label=False):
    """Calculate Mean Dice (mDice)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.

     Returns:
        dict[str, float | ndarray]: Default metrics.
            <aAcc> float: Overall accuracy on all images.
            <Acc> ndarray: Per category accuracy, shape (num_classes, ).
            <Dice> ndarray: Per category dice, shape (num_classes, ).
    """

    dice_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mDice'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return dice_result


def mean_fscore(results,
                gt_seg_maps,
                num_classes,
                ignore_index,
                nan_to_num=None,
                label_map=dict(),
                reduce_zero_label=False,
                beta=1):
    """Calculate Mean Intersection and Union (mIoU)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.
        beta (int): Determines the weight of recall in the combined score.
            Default: False.


     Returns:
        dict[str, float | ndarray]: Default metrics.
            <aAcc> float: Overall accuracy on all images.
            <Fscore> ndarray: Per category recall, shape (num_classes, ).
            <Precision> ndarray: Per category precision, shape (num_classes, ).
            <Recall> ndarray: Per category f-score, shape (num_classes, ).
    """
    fscore_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mFscore'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label,
        beta=beta)
    return fscore_result


def eval_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 ignore_index,
                 metrics=['mIoU'],
                 nan_to_num=None,
                 label_map=dict(),
                 reduce_zero_label=False,
                 beta=1):
    """Calculate evaluation metrics
    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str] | Iterables): list of ground
            truth segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """

    total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label = total_intersect_and_union(
            results, gt_seg_maps, num_classes, ignore_index, label_map,
            reduce_zero_label)
    ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union,
                                        total_area_pred_label,
                                        total_area_label, metrics, nan_to_num,
                                        beta)

    return ret_metrics


def pre_eval_to_metrics(pre_eval_results,
                        metrics=['mIoU'],
                        nan_to_num=None,
                        beta=1,
                        mean=True):
    """Convert pre-eval results to metrics.

    Args:
        pre_eval_results (list[tuple[torch.Tensor]]): per image eval results
            for computing evaluation metric
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """

    # convert list of tuples to tuple of lists, e.g.
    # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
    # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
    pre_eval_results = tuple(zip(*pre_eval_results))
    assert len(pre_eval_results) == 4

    if mean:
        total_area_intersect = sum(pre_eval_results[0])
        total_area_union = sum(pre_eval_results[1])
        total_area_pred_label = sum(pre_eval_results[2])
        total_area_label = sum(pre_eval_results[3])
    else:
        # ([A_1, ..., A_n], ..., [D_1, ..., D_n]) to
        # (torch.tensor([A_1, ..., A_n]), ..., torch.tensor([D_1, ..., D_n]))
        total_area_intersect = torch.stack(pre_eval_results[0])
        total_area_union = torch.stack(pre_eval_results[1])
        total_area_pred_label = torch.stack(pre_eval_results[2])
        total_area_label = torch.stack(pre_eval_results[3])

    ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union,
                                        total_area_pred_label,
                                        total_area_label, metrics, nan_to_num,
                                        beta,
                                        mean=mean)

    return ret_metrics


def total_area_to_metrics(total_area_intersect,
                          total_area_union,
                          total_area_pred_label,
                          total_area_label,
                          metrics=['mIoU'],
                          nan_to_num=None,
                          beta=1,
                          mean=True):
    """Calculate evaluation metrics
    Args:
        total_area_intersect (ndarray): The intersection of prediction and
            ground truth histogram on all classes.
        total_area_union (ndarray): The union of prediction and ground truth
            histogram on all classes.
        total_area_pred_label (ndarray): The prediction histogram on all
            classes.
        total_area_label (ndarray): The ground truth histogram on all classes.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['mIoU', 'mDice', 'mFscore', 'IoU+f1']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))

    if mean:
        all_acc = total_area_intersect.sum() / total_area_label.sum()
    else:
        all_acc = total_area_intersect.sum(dim=0).sum() / total_area_label.sum(dim=0).sum()
    ret_metrics = OrderedDict({'aAcc': all_acc})
    for metric in metrics:
        if metric == 'mIoU':
            iou = total_area_intersect / total_area_union
            acc = total_area_intersect / total_area_label
            ret_metrics['IoU'] = iou
            ret_metrics['Acc'] = acc
        elif metric == 'mDice':
            dice = 2 * total_area_intersect / (
                total_area_pred_label + total_area_label)
            acc = total_area_intersect / total_area_label
            ret_metrics['Dice'] = dice
            ret_metrics['Acc'] = acc
        elif metric == 'mFscore':
            if not mean:
                precision = total_area_intersect / total_area_pred_label
                recall = total_area_intersect / total_area_label
                f_value = torch.stack(
                    [f_score(x[0], x[1], beta) for x in zip(precision, recall)])
                # precision_mean = total_area_intersect.sum(dim=0) / total_area_pred_label.sum(dim=0)
                # recall_mean = total_area_intersect.sum(dim=0) / total_area_label.sum(dim=0)
                # f_value = torch.tensor(
                #         [f_score(x[0], x[1], beta) for x in zip(precision_mean, recall_mean)])
                
                ret_metrics['Fscore'] = torch.from_numpy(np.nanmean(f_value.cpu().numpy(), 
                                                                    axis=0)).to(precision.device)
                # ret_metrics['Precision'] = precision
                # ret_metrics['Recall'] = recall
                ret_metrics['Precision'] = torch.from_numpy(np.nanmean(precision.cpu().numpy(), 
                                                                    axis=0)).to(precision.device)
                ret_metrics['Recall'] = torch.from_numpy(np.nanmean(recall.cpu().numpy(), 
                                                                    axis=0)).to(precision.device)
            else:
                precision = total_area_intersect / total_area_pred_label
                recall = total_area_intersect / total_area_label
                f_value = torch.tensor(
                    [f_score(x[0], x[1], beta) for x in zip(precision, recall)])
                ret_metrics['Fscore'] = f_value
                ret_metrics['Precision'] = precision
                ret_metrics['Recall'] = recall
        elif metric == 'IoU+f1':
            iou = total_area_intersect / total_area_union
            iou_mean = total_area_intersect.sum(dim=0) / total_area_union.sum(dim=0)

            precision = total_area_intersect / total_area_pred_label
            recall = total_area_intersect / total_area_label

            precision_mean = total_area_intersect.sum(dim=0) / total_area_pred_label.sum(dim=0)
            recall_mean = total_area_intersect.sum(dim=0) / total_area_label.sum(dim=0)

            f_value = torch.stack(
                [f_score(x[0], x[1], beta) for x in zip(precision, recall)]).nansum(dim=0)
            iou = iou.nansum(dim=0)

            ret_metrics['Fscore'] = torch.tensor(
                    [f_score(x[0], x[1], beta) for x in zip(precision_mean, recall_mean)])
            ret_metrics['IoU'] = iou_mean
            ret_metrics['IoU+f1'] = f_value + iou

    ret_metrics = {
        metric: value.numpy()
        for metric, value in ret_metrics.items()
    }
    if nan_to_num is not None:
        ret_metrics = OrderedDict({
            metric: np.nan_to_num(metric_value, nan=nan_to_num)
            for metric, metric_value in ret_metrics.items()
        })
    return ret_metrics


class EvalHook(_EvalHook):
    """Single GPU EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
        pre_eval (bool): Whether to use progressive mode to evaluate model.
            Default: False.
    Returns:
        list: The prediction results.
    """

    greater_keys = ['mIoU', 'mAcc', 'aAcc', 'IoU+f1']

    def __init__(self,
                 *args,
                 by_epoch=False,
                 efficient_test=False,
                 pre_eval=False,
                 **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.pre_eval = pre_eval
        if efficient_test:
            warnings.warn(
                'DeprecationWarning: ``efficient_test`` for evaluation hook '
                'is deprecated, the evaluation hook is CPU memory friendly '
                'with ``pre_eval=True`` as argument for ``single_gpu_test()`` '
                'function')

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        from .test import single_gpu_test
        results = single_gpu_test(
            runner.model, self.dataloader, show=False, pre_eval=self.pre_eval)
        runner.log_buffer.clear()
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        if self.save_best:
            self._save_ckpt(runner, key_score)


class DistEvalHook(_DistEvalHook):
    """Distributed EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
        pre_eval (bool): Whether to use progressive mode to evaluate model.
            Default: False.
    Returns:
        list: The prediction results.
    """

    greater_keys = ['mIoU', 'mAcc', 'aAcc', 'IoU+f1']

    def __init__(self,
                 *args,
                 by_epoch=False,
                 efficient_test=False,
                 pre_eval=False,
                 thresh=0.5,
                 **kwargs):
        super().__init__(*args, by_epoch=by_epoch, thresh=thresh, **kwargs)
        self.pre_eval = pre_eval
        self.thresh = thresh
        if efficient_test:
            warnings.warn(
                'DeprecationWarning: ``efficient_test`` for evaluation hook '
                'is deprecated, the evaluation hook is CPU memory friendly '
                'with ``pre_eval=True`` as argument for ``multi_gpu_test()`` '
                'function')

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect,
            pre_eval=self.pre_eval,
            thresh=self.thresh)

        runner.log_buffer.clear()

        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)

            if self.save_best:
                self._save_ckpt(runner, key_score)


def _format_print_results(cate_name, results, dataset_name, metric='mFscore', 
                    mean=False, CLASSES=('authentic', 'forgery'), binary_img_gt=True, **kwargs):
    ret_metrics = {}
    ious = [result['iou'] for result in results if 'iou' in result and result['iou'] is not None]
    if len(ious) > 0:
        pixel_metrics1 = pre_eval_to_metrics(ious, metric, mean=mean)
        ret_metrics.update(pixel_metrics1)

    aucs = [result['auc'] for result in results if 'auc' in result and result['auc'] is not None]
    if len(aucs) > 0:
        ret_metrics['pixel_auc'] = np.array(aucs)

    img_eval = [result['img_eval'] for result in results if 'img_eval' in result and result['img_eval'] is not None]
    gts = np.array([item[0] for item in img_eval]).astype(np.int32).flatten()
    if binary_img_gt:
        gts[gts >=1 ] = 1
    scores = np.array([item[1] for item in img_eval])
    if len(img_eval) > 0 and gts.min() == 0:
        try:
            img_auc = calculate_auc(scores, gts)
        except ValueError:
            img_auc = 0.0
        ret_metrics['img_auc'] = img_auc

        thresh = kwargs.get('img_thresh', kwargs.get('thresh', None))

        pred = (scores > thresh).astype(np.int32).flatten()
        img_f1, img_precision, img_recall = calculate_img_score(pred, gts)[0:3]
        ret_metrics['img_f1'] = img_f1
        ret_metrics['img_p'] = img_precision
        ret_metrics['img_r'] = img_recall

        thresh = kwargs.get('img_thresh', thresh)
        img_acc = (pred == gts).sum() / pred.shape[0]
        ret_metrics['img_acc'] = img_acc
    elif len(img_eval) > 0:
        thresh = kwargs.get('img_thresh', kwargs.get('thresh', 0.5))
        pred = (scores > thresh).astype(np.int32).flatten()
        img_acc = (pred == gts).sum() / pred.shape[0]
        ret_metrics['img_acc'] = img_acc

    # Because dataset.CLASSES is required for per-eval.
    class_names = CLASSES

    # summary table
    ret_metrics_summary = OrderedDict({
        ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2) \
                    if ret_metric not in ('IoU+f1', 'pixel_auc', 'img_auc') \
                        else np.round(np.nanmean(ret_metric_value), 3)
        for ret_metric, ret_metric_value in ret_metrics.items()
    })

    # each class table
    ret_metrics.pop('aAcc', None)
    ret_metrics.pop('img_f1', None)
    ret_metrics.pop('img_auc', None)
    ret_metrics.pop('img_p', None)
    ret_metrics.pop('img_r', None)
    ret_metrics.pop('img_acc', None)
    ret_metrics.pop('pixel_auc', None)
    ret_metrics_class = OrderedDict({
        ret_metric: np.round(ret_metric_value * 100, 2) \
                    if ret_metric not in {'IoU+f1', 'pixel_auc', 'img_auc'} \
                        else np.round(ret_metric_value, 3)
        for ret_metric, ret_metric_value in ret_metrics.items()
    })
    ret_metrics_class.update({'Class': class_names})
    ret_metrics_class.move_to_end('Class', last=False)

    # for logger
    class_table_data = PrettyTable()
    for key, val in ret_metrics_class.items():
        class_table_data.add_column(key, val)

    summary_table_data = PrettyTable()
    for key, val in ret_metrics_summary.items():
        if key in {'aAcc', 'IoU+f1', 'img_f1', 'img_auc', 'img_p', 'img_r',
                'pixel_auc', 'img_acc'}:
            summary_table_data.add_column(key, [val])
        else:
            summary_table_data.add_column('m' + key, [val])

    if get_dist_info()[0] == 0:
        print_dataset_name = f'Category {cate_name} results in Dataset {dataset_name} '
        print_log('{}per class results:'.format(print_dataset_name), get_root_logger())
        print_log('\n' + class_table_data.get_string(), logger=get_root_logger())
        print_log('Category {} {} Summary:'.format(cate_name, dataset_name), logger=get_root_logger())
        print_log('\n' + summary_table_data.get_string(), logger=get_root_logger())


# _forge_cates = ['Au', 'Copy-move', 'Splicing', 'Inpainting', 
#                 'small', 'median', 'large']
# _forge_cates = _forge_cates + list(map(lambda item: item[0]+'-'+item[1], 
#                                         product(_forge_cates[1:4], _forge_cates[4:])))
_forge_cates = ['Au', 'Inpainting', 'Splicing', 'Copy-move', 'Text', 'rand_text', 'rand_splicing', 'seal_splicing', 'seal_rand_splicing',
                'small', 'median', 'large']
_forge_cates = _forge_cates + list(map(lambda item: item[0]+'-'+item[1], 
                                        product(_forge_cates[1:7], _forge_cates[7:])))



def dataloader_test(test_dataloader, model, cfgs, 
                    sizes_ratios=(0.05, 0.05)):
    model.eval()

    dataset = test_dataloader.dataset
    dataset_name = dataset.dataset_name
    if 'ext_test_dataset' not in cfgs or dataset_name not in cfgs.ext_test_dataset:
        return
    dataset.gt_seg_map_loader.img_label_binary = False
    thresh = cfgs.evaluation.thresh
    loader_indices = test_dataloader.batch_sampler

    rank, _ = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    results_per_cates = defaultdict(list)
    for batch_indices, inputs in zip(loader_indices, test_dataloader):
        # img is instance of DataContainer
        img = inputs['img'].data.cuda()
        img_meta = inputs['img_metas'].data[0]
        if 'dct_vol' in inputs:
            dct_vol = inputs['dct_vol'].data.cuda()
            qtables = inputs['qtables'].data.cuda()
        # img_name = img_meta[0]['filename']
        mask_filename = img_meta[0]['mask_filename']

        with torch.no_grad():
            if 'dct_vol' in inputs:
                dct_vol = dct_vol.unsqueeze(dim=0)
                qtables = qtables.unsqueeze(dim=0)
                result = model(img, dct_vol, qtables, [img_meta, ], return_loss=False, rescale=True)[1][0, 0]
            else:
                result = model(img, img_meta, return_loss=False, rescale=True)[1][0, 0]
        
        result = dataset.pre_eval(result, indices=batch_indices, thresh=thresh)[0]
        # pixel_iou = result['iou']
        # pixel_auc = result['auc']
        # img_gt, img_score = result['img_eval']
        cate_name = _forge_cates[result['img_eval'][0]]
        results_per_cates[cate_name].append(result)

        if osp.isfile(mask_filename):
            mask_im = cv2.imread(mask_filename)
            if mask_im is not None and mask_im.min() == 0 and mask_im.max() == 255:
                mask_im[mask_im >= 127] = 255
                mask_im[mask_im < 127] = 0
                ratio = mask_im[mask_im == 255].size / mask_im.size

                if ratio <= sizes_ratios[0]:
                    ratio_name = 'small'
                elif sizes_ratios[0] < ratio <= sizes_ratios[1]:
                    ratio_name = 'median'
                else:
                    ratio_name = 'large'

                results_per_cates[ratio_name].append(result)
                results_per_cates[cate_name+'-'+ratio_name].append(result)

        if rank == 0:
            prog_bar.update()

    # for i, cate_result in enumerate(results_per_cates[1:]):
    for name, cate_result in results_per_cates.items():
        if name == 'Au':
            continue
        _format_print_results(name, 
                            cate_result, 
                            dataset_name, 
                            # metric='mFscore', 
                            # mean=False, 
                            CLASSES=('authentic', 'forgery'), 
                            **cfgs.evaluation)