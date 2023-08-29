import os.path as osp
import warnings
import random
from collections import OrderedDict
import mmcv
import os
import numpy as np
import copy
import cv2
from sklearn import metrics
from mmcv.utils import print_log
from prettytable import PrettyTable
from torch.utils.data import Dataset
import pytesseract as pt
from .pipelines import Compose, LoadAnnotations
from ..utils.logger import get_root_logger
from ..utils.evaluate import (intersect_and_union, eval_metrics, pre_eval_to_metrics,
                            calculate_auc, calculate_img_score)
from .builder import DATASETS
from .pipelines import gen_high_freq


@DATASETS.register_module()
class MaskSegDataset(Dataset):
    """Custom dataset for semantic segmentation. An example of file structure
    is as followed.

    .. code-block:: none

        ├── data
        │   ├── my_dataset
        │   │   ├── img_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── val
        │   │   ├── ann_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{seg_map_suffix}
        │   │   │   │   ├── yyy{seg_map_suffix}
        │   │   │   │   ├── zzz{seg_map_suffix}
        │   │   │   ├── val

    The img/gt_semantic_seg pair of CustomDataset should be of the same
    except suffix. A valid img/gt_semantic_seg filename pair should be like
    ``xxx{img_suffix}`` and ``xxx{seg_map_suffix}`` (extension is also included
    in the suffix). If split is given, then ``xxx`` is specified in txt file.
    Otherwise, all files in ``img_dir/``and ``ann_dir`` will be loaded.
    Please refer to ``docs/en/tutorials/new_dataset.md`` for more details.


    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        img_suffix (str): Suffix of images. Default: '.jpg'
        ann_dir (str, optional): Path to annotation directory. Default: None
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_dir/ann_dir will be loaded. Default: None
        data_root (str, optional): Data root for img_dir/ann_dir. Default:
            None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        ignore_index (int): The label index to be ignored. Default: 255
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default: False
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        palette (Sequence[Sequence[int]]] | np.ndarray | None):
            The palette of segmentation map. If None is given, and
            self.PALETTE is None, random palette will be generated.
            Default: None
        gt_seg_map_loader_cfg (dict, optional): build LoadAnnotations to
            load gt for evaluation, load from disk by default. Default: None.
    """

    CLASSES = ['medium', 'collagen']

    PALETTE = [0, 1]

    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix='.jpg',
                 ann_dir=None,
                 seg_map_suffix='.png',
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index=None,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None,
                 gt_seg_map_loader_cfg=None,
                 simulate_p=0.5,
                 text_det=None,
                 dataset_name='',
                 **unused_kwargs):
        self.pre_pipelines = None
        if mmcv.is_list_of(pipeline, list):
            self.pre_pipelines = Compose(pipeline[0])
            self.post_pipelines = Compose(pipeline[1])
        else:
            self.post_pipelines = Compose(pipeline)
        # if simulate_aug:
        #     self.text_det_results = {}
        #     if text_det is not None:
        #         if img_dir is not None:
        #             text_det = osp.join(data_root, text_det)
        #         with open(text_det) as f:
        #             self.text_det_results = json.load(f)

        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)
        self.gt_seg_map_loader = LoadAnnotations(binary=True
        ) if gt_seg_map_loader_cfg is None else LoadAnnotations(
            **gt_seg_map_loader_cfg)
        # self.simulate_aug = simulate_aug
        self.simulate_p = simulate_p
        self.dataset_name = dataset_name

        if test_mode:
            assert self.CLASSES is not None, \
                '`cls.CLASSES` or `classes` should be specified when testing'

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        # load annotations
        self.img_infos = self.load_annotations(self.img_dir, self.img_suffix,
                                               self.ann_dir,
                                               self.seg_map_suffix, self.split)

    def __len__(self):
        """Total number of samples of data."""
        # if self.test_mode:
        #     return 50
        return len(self.img_infos)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
            img_infos = sorted(img_infos, key=lambda x: x['filename'])

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']

    def get_img_info(self, idx):
        return self.img_infos[idx]

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        # if self.custom_classes:
        #     results['label_map'] = self.label_map

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        # results contents:
        # { 'img_info': img_info,
        #   'ann_info': ann_info,
        #   'seg_fields': [],
        #   'img_prefix': self.img_dir,
        #   'seg_prefix': self.ann_dir,
        #   'filename': ,
        #   'ori_filename': ,
        #   'img': img,
        #   'img_shape':  ,
        #   'ori_shape': img.shape,
        #   'pad_shape': img.shape,
        #   'scale_factor': 1.0,
        #   'img_norm_cfg': ,
        #   'gt_semantic_seg': gt_semantic_seg,
        #   'seg_fields': ['gt_semantic_seg']
        # }
        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            ret = self.prepare_train_img(idx)
            # print(ret)
            return ret


    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        if self.pre_pipelines:
            results = self.pre_pipelines(results)
            # results = self.multi_rand_aug(results)
        results = self.post_pipelines(results)
        return results

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """

        try:
            img_info = self.img_infos[idx]
            results = dict(img_info=img_info)
            self.pre_pipeline(results)
            if self.pre_pipelines:
                results = self.pre_pipelines(results)
            results = self.post_pipelines(results)
        except Exception as e:
            raise e
        return results

    # def format_results(self, results, imgfile_prefix, indices=None, **kwargs):
    #     """Place holder to format result to dataset specific output."""
    #     raise NotImplementedError

    def get_gt_seg_map_by_idx(self, index):
        """Get one ground truth segmentation map for evaluation."""
        ann_info = self.get_ann_info(index)
        results = dict(ann_info=ann_info)
        self.pre_pipeline(results)
        self.gt_seg_map_loader(results)
        # return results['gt_semantic_seg']
        return results

    def get_gt_seg_maps(self, efficient_test=None):
        """Get ground truth segmentation maps for evaluation."""
        if efficient_test is not None:
            warnings.warn(
                'DeprecationWarning: ``efficient_test`` has been deprecated '
                'since MMSeg v0.16, the ``get_gt_seg_maps()`` is CPU memory '
                'friendly by default. ')

        for idx in range(len(self)):
            ann_info = self.get_ann_info(idx)
            results = dict(ann_info=ann_info)
            self.pre_pipeline(results)
            self.gt_seg_map_loader(results)
            yield results['gt_semantic_seg']

    def pre_eval(self, preds, indices, thresh=None):
        """Collect eval result from each iteration.

        Args:
            preds (list[torch.Tensor] | torch.Tensor): the segmentation logit
                after argmax, shape (N, H, W).
            indices (list[int] | int): the prediction related ground truth
                indices.

        Returns:
            list[torch.Tensor]: (area_intersect, area_union, area_prediction,
                area_ground_truth).
        """
        # In order to compat with batch inference
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]

        pre_eval_results = []
        for pred, index in zip(preds, indices):
            if not isinstance(pred, tuple):
                seg_pred = pred
                cls_pred = None
            else:
                cls_pred, seg_pred = pred
                # 1 is positive
                if seg_pred.shape[1] > 1:
                    seg_pred = seg_pred[:, 1, ...]
            # seg_map = self.get_gt_seg_map_by_idx(index).astype(pred.dtype)
            annos = self.get_gt_seg_map_by_idx(index)
            seg_gt = annos.get('gt_semantic_seg', None)
            if seg_gt is not None:
                seg_gt = seg_gt.astype(seg_pred.dtype)
            img_gt = annos['img_label']

            # if img_gt > 0 and seg_gt.max() > 0:
            if img_gt > 0:
                # if seg_gt.max() == 0.:
                    # print(self.img_infos[index]['ann']['seg_map'])

                if thresh is None:
                    seg_pred_ = seg_pred.flatten()
                    seg_gt_ = seg_gt.astype(np.int).flatten()

                    tpr, fpr, thresholds = metrics.roc_curve(seg_gt_, seg_pred_, pos_label=1)
                    max_index = (tpr-fpr).argmax()
                    thresh = thresholds[max_index]

                pred_l = (seg_pred > thresh).astype(np.float32)

                iou = intersect_and_union(pred_l, 
                                         seg_gt, 
                                         len(self.CLASSES),
                                         self.ignore_index, 
                                         self.label_map,
                                         self.reduce_zero_label)
                try:
                    auc = calculate_auc(seg_pred.flatten(), seg_gt.flatten())
                except ValueError as e:
                    auc = None
            else:
                iou = None
                auc = None

            score = np.max(seg_pred) if cls_pred is None else cls_pred
            img_eval = [img_gt, score]

            pre_eval_results.append({'iou': iou, 'auc': auc, 'img_eval': img_eval})

        return pre_eval_results

    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Default: None
        """
        return self.CLASSES, self.PALETTE

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 gt_seg_maps=None,
                 mean=False,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                 results or predict segmentation map for computing evaluation
                 metric.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            gt_seg_maps (generator[ndarray]): Custom gt seg maps as input,
                used in ConcatDataset

        Returns:
            dict[str, float]: Default metrics.
        """
        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore', 'IoU+f1']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))

        # test a list of files
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(
                results, str):
            if gt_seg_maps is None:
                gt_seg_maps = self.get_gt_seg_maps()
            num_classes = len(self.CLASSES)
            ret_metrics = eval_metrics(
                results,
                gt_seg_maps,
                num_classes,
                self.ignore_index,
                metric,
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label)
        # test a list of pre_eval_results
        else:
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
            scores = np.array([item[1] for item in img_eval])
            if len(img_eval) > 0 and gts.min() == 0:
                img_auc = calculate_auc(scores, gts)
                ret_metrics['img_auc'] = img_auc

                thresh = kwargs.get('img_thresh', kwargs.get('thresh', None))
                if thresh is None:
                    tpr, fpr, thresholds = metrics.roc_curve(gts, scores, pos_label=1)
                    max_index = (tpr-fpr).argmax()
                    thresh = thresholds[max_index]

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
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

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

        print_dataset_name = 'Dataset ' + self.dataset_name+' ' if self.dataset_name else ''
        print_log('{}per class results:'.format(print_dataset_name), logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        print_log('{}Summary:'.format(print_dataset_name), logger)
        print_log('\n' + summary_table_data.get_string(), logger=logger)

        eval_results = {}
        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key in {'aAcc', 'pixel_auc'}:
                eval_results[key] = value / 100.0
            elif key in {'IoU+f1'}:
                eval_results[key] = value
            else:
                eval_results['m' + key] = value / 100.0

        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): (value[idx] / 100.0) if key != 'IoU+f1' else value[idx]
                for idx, name in enumerate(class_names)
            })

        log_dataset_name = self.dataset_name+'/' if self.dataset_name else ''
        eval_results_ = eval_results
        eval_results = {}
        for k, v in eval_results_.items():
            eval_results[log_dataset_name+k] = v

        return eval_results

    def detect_text_bboxes(self, img):
        text_bboxes = []

        tess_text = pt.image_to_data(img, lang='chi_sim', output_type=pt.Output.DICT, config='--oem 1 --psm 6 words')
        for i in range(len(tess_text['text'])):
            # conf = tess_text['conf'][i]
            word_len = len(tess_text['text'][i])
            # if word_len > 1 and conf >= 10:
            if word_len > 1:
                x, y, w, h = tess_text['left'][i], tess_text['top'][i], tess_text['width'][i], tess_text['height'][i]
                if w > 0 and h > 0:
                    text_bboxes.append((x, y, x + w, y + h))

        return text_bboxes

    def _choose_box(self, mask, text_bboxes):
        retries = 30
        while retries > 0:
            bbox = random.choice(text_bboxes)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            tmp_mask = np.zeros(mask.shape)
            tmp_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
            if (mask * tmp_mask).sum() == 0:
                return bbox

            retries -= 1

        return None

    def _do_mosaic(self, img, x, y, w, h, neighbors):
        if neighbors[0] <= 0 or neighbors[1] <= 0:
            return
        w_n, h_n = neighbors
        for i in range(0, h - h_n, h_n):
            for j in range(0, w - w_n, w_n):
                rect = [j + x, i + y, w_n, h_n]
                color = img[i + y, j + x].tolist()
                left_up = (rect[0], rect[1])
                right_down = (rect[0] + w_n - 1, rect[1] + h_n - 1)
                cv2.rectangle(img, left_up, right_down, color, -1)

    def _choose_proper_box(self, src_bboxes, target_bbox, mul_ratio=3.0, mul_wh = 2.0):
        proper_bboxes = []

        t_w, t_h = target_bbox[2]-target_bbox[0], target_bbox[3]-target_bbox[1]
        t_ratio = np.sqrt(t_w * t_h)
        for s_bbox in src_bboxes:
            w, h = s_bbox[2] - s_bbox[0], s_bbox[3] - s_bbox[1]
            ratio = np.sqrt(w * h)
            if max(ratio, t_ratio)/min(ratio, t_ratio) <= mul_ratio \
                and max(t_w, w)/min(t_w, w) <= mul_wh \
                and max(t_h, h)/min(t_h, h) <= mul_wh:
                proper_bboxes.append(s_bbox)

        if len(proper_bboxes) == 0:
            return None
        return random.choice(proper_bboxes)

    def _random_bbox(self, img_h, img_w, min_p=0.05, max_p=0.15):
        r_h = random.randint(int(min_p*img_h), int(max_p*img_h))
        r_w = random.randint(int(min_p*img_w), int(max_p * img_w))
        r_y = random.randint(0, img_h-r_h)
        r_x = random.randint(0, img_w - r_w)

        return r_x, r_y, r_w+r_x, r_h+r_y

    def multi_rand_aug(self, results):
        dst_mask = results['gt_semantic_seg']
        all_neg = dst_mask[dst_mask == 1].size == 0

        if random.random() > self.simulate_p and not all_neg:
            return results

        aug_num = 2
        if all_neg:
            aug_num = 4

        for _ in range(random.randint(1, aug_num)):
            results = self.rand_aug_v3(results, all_neg)

        return results

    RAND_AUG_POLICY = ('text_removal', 'text_splicing', 'random_forgery')
    REMOVAL_POLICY = ('mosaic', 'remove', 'copy_move', 'inpaint')

    RAND_AUG_POLICY_v2 = ('rand_copy_move', 'text_splicing' , 'text_mosaic', 'text_remove', 'text_copy_move', 'text_inpaint')

    def rand_aug_v3(self, results, all_neg):
        dst_img = results['img']
        dst_mask = results['gt_semantic_seg']
        filename = results['filename']
        im_h, im_w = dst_img.shape[0:2]

        policy = random.choice(self.RAND_AUG_POLICY_v2)
        if 'text' in policy:
            if not all_neg and filename in self.text_det_results:
                dst_text_bboxes = self.text_det_results[filename]
            else:
                dst_text_bboxes = self.detect_text_bboxes(dst_img)
                if not all_neg:
                    self.text_det_results[filename] = dst_text_bboxes

            if len(dst_text_bboxes) == 0:
                return results

            # bbox: x1,y1,x2,y2
            bbox = self._choose_box(dst_mask, dst_text_bboxes)
            if bbox is None:
                return results

            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x1, y1, x2, y2 = bbox

            if policy == 'text_mosaic':
                # fill with mosaic content
                self._do_mosaic(dst_img, x1, y1, w, h, neighbors=(w // 5, h // 5))
                dst_mask[y1:y2, x1:x2] = 1
            elif policy == 'text_remove':
                if len(dst_img.shape) == 2:
                    size = (h, w)
                else:
                    size = (h, w, 3)
                # fill with random content
                dst_img[y1:y2, x1:x2, ...] = np.random.uniform(200, 255, size)
                dst_mask[y1:y2, x1:x2] = 1
            elif policy == 'text_copy_move':
                r_y = random.randint(0, im_h - h)
                r_x = random.randint(0, im_w - w)

                dst_img[r_y:r_y + h, r_x:r_x + w] = dst_img[y1:y2, x1:x2]
                dst_mask[r_y:r_y + h, r_x:r_x + w] = 1
            elif policy == 'text_inpaint':
                y_min = max(y1 - 10, 0)
                x_min = max(x1 - 10, 0)
                y_max = min(y2 + 10, im_h)
                x_max = min(x2 + 10, im_w)
                h_min = max(0, y1 - y_min)
                w_min = max(0, x1 - x_min)
                # 使用cv2.inpaint方法，提高真实度，作为可选
                mask = np.zeros((y_max - y_min, x_max - x_min)).astype(np.uint8)
                mask[h_min:h_min + h, w_min:w_min + w] = 1
                inpainted_block = cv2.inpaint(dst_img[y_min:y_max, x_min:x_max], mask, 5, cv2.INPAINT_NS)
                dst_img[y1:y2, x1:x2] = inpainted_block[h_min:h + h_min, w_min:w + w_min]
                dst_mask[y1:y2, x1:x2] = 1
            elif policy == 'text_splicing':
                retries = 50
                while retries > 0:
                    src_idx = random.randint(0, len(self) - 1)
                    img_info = self.img_infos[src_idx]
                    filename = img_info['filename']
                    if self.img_dir:
                        filename = os.path.join(self.img_dir, filename)

                    if filename in self.text_det_results:
                        src_bboxes = self.text_det_results[filename]
                        src_im = None
                    else:
                        src_im = cv2.imread(filename)
                        src_bboxes = self.detect_text_bboxes(src_im)
                        results[filename] = src_bboxes

                        self.text_det_results[filename] = dst_text_bboxes

                    if len(src_bboxes) == 0:
                        retries -= 1
                        continue

                    src_box = self._choose_proper_box(src_bboxes, bbox)
                    if src_box is not None:
                        break

                    retries -= 1
                else:
                    return results

                if src_im is None:
                    src_im = cv2.imread(filename)
                src_img_region = src_im[src_box[1]:src_box[3], src_box[0]:src_box[2]]

                if random.random() < 0.5:
                    # 使用cv2.inpaint方法，提高真实度，作为可选
                    y_min = max(y1 - 10, 0)
                    x_min = max(x1 - 10, 0)
                    y_max = min(y2 + 10, im_h)
                    x_max = min(x2 + 10, im_w)
                    h_min = max(0, y1 - y_min)
                    w_min = max(0, x1 - x_min)
                    # 使用cv2.inpaint方法，提高真实度，作为可选
                    mask = np.zeros((y_max - y_min, x_max - x_min)).astype(np.uint8)
                    mask[h_min:h_min + h, w_min:w_min + w] = 1
                    inpainted_block = cv2.inpaint(dst_img[y_min:y_max, x_min:x_max], mask, 5, cv2.INPAINT_NS)
                    dst_img[y1:y2, x1:x2] = inpainted_block[h_min:h + h_min, w_min:w + w_min]

                if random.random() < 0.5:
                    # 只提取src_img_region文字纹理，覆盖到dst_img上，作为可选
                    gray = cv2.cvtColor(src_img_region, cv2.COLOR_BGR2GRAY)
                    _, text_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    text_mask[text_mask == 255] = 1

                    src_img_region = cv2.resize(src_img_region, (w, h))
                    text_mask = cv2.resize(text_mask, (w, h))

                    for i in range(dst_img.shape[-1]):
                        dst_img_block = dst_img[y1:y2, x1:x2, i]
                        dst_img_block[text_mask == 1] = src_img_region[:, :, i][text_mask == 1]
                        dst_img[y1:y2, x1:x2, i] = dst_img_block

                    dst_mask[y1:y2, x1:x2] = text_mask
                else:
                    dst_img[y1:y2, x1:x2] = cv2.resize(src_img_region, (w, h))
                    dst_mask[y1:y2, x1:x2] = 1

        elif policy == 'rand_copy_move':
            # r_policy = random.choice(self.REMOVAL_POLICY)
            # r_policy = random.choice(('copy_move',))
            # r_policy = 'copy_move'

            x1, y1, x2, y2 = self._random_bbox(im_h, im_w)
            w, h = x2 - x1, y2 - y1

            # copy_move
            r_y = random.randint(0, im_h - h)
            r_x = random.randint(0, im_w - w)
            dst_img[r_y:r_y + h, r_x:r_x + w] = dst_img[y1:y2, x1:x2]
            dst_mask[r_y:r_y + h, r_x:r_x + w] = 1

        return results


@DATASETS.register_module()
class MaskSegDatasetv2(MaskSegDataset):
    def __init__(self,
                 pipeline,
                 data_root,
                 ann_path,
                 edge_mask_dir=None,
                 test_mode=False,
                 ignore_index=None,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None,
                 gt_seg_map_loader_cfg=None,
                 simulate_p=0.0,
                 dataset_name=''):
        self.pipeline_cfg = pipeline
        self.pre_pipelines = None
        if mmcv.is_list_of(pipeline, list):
            self.pre_pipelines = Compose(pipeline[0])
            self.post_pipelines = Compose(pipeline[1])
        else:
            self.post_pipelines = Compose(pipeline)

        self.data_root = data_root
        self.ann_path = ann_path
        self.edge_mask_dir = edge_mask_dir
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)
        self.gt_seg_map_loader = LoadAnnotations(binary=True
        ) if gt_seg_map_loader_cfg is None else LoadAnnotations(
            **gt_seg_map_loader_cfg)
        self.simulate_p = simulate_p
        self.dataset_name = dataset_name

        if test_mode:
            assert self.CLASSES is not None, \
                '`cls.CLASSES` or `classes` should be specified when testing'

        # load annotations
        self.img_infos = self.load_annotations()

    def load_annotations(self):
        img_infos = []

        anno_path = osp.join(self.data_root, self.ann_path)
        with open(anno_path) as f:
            lines = [line.strip() for line in f.readlines()]
    
        for line in lines:
            try:
                img_path, gt_path_= line.split(' ')
            except Exception as e:
                print(line)
                raise e

            img_path = osp.join(self.data_root, img_path)
            gt_path = osp.join(self.data_root, gt_path_)

            if not osp.isfile(img_path):
                continue
            if gt_path_ != 'None' and not osp.isfile(gt_path):
                continue

            img_info = dict(filename=img_path)
            img_info['ann'] = dict(seg_map=gt_path)
            # img_info['ann']['img_label'] = int(label)
            if self.edge_mask_dir:
                img_info['ann']['edge_mask'] = osp.join(self.edge_mask_dir, osp.basename(gt_path))
            img_infos.append(img_info)                

        # img_infos = sorted(img_infos, key=lambda x: x['filename'])

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = None
        results['seg_prefix'] = None

    def multi_rand_aug(self, results):
        mask = results['gt_semantic_seg']

        if (mask==1).any() or random.random() > self.simulate_p:
            return results

        results = self.rand_aug_v4(results)

        return results

    @staticmethod
    def gen_edge(gt, epochs=12, kernel_size=8):
        gt = copy.deepcopy(gt)
        gt[gt==1] = 255

        kernel = np.ones((kernel_size, kernel_size), np.uint8) 
        gt = cv2.dilate(gt, kernel)

        edge_mask = np.zeros(gt.shape)
        for _ in range(epochs):
            # 形态学：边缘检测
            _,Thr_img = cv2.threshold(gt,128,255,cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
            gradient = cv2.morphologyEx(Thr_img, cv2.MORPH_GRADIENT, kernel)
            edge_mask[gradient==255] = 255
            gt[gradient==255] = 0
        edge_mask[edge_mask==255] = 1

        return edge_mask

    @staticmethod
    def generate_mask(img_height,img_width,radius1, radius2, center_x, center_y):
        y,x = np.ogrid[0:img_height,0:img_width]
        mask = ((x-center_x)/radius1)**2+((y-center_y)/radius2)**2<=1
        return mask

    def rand_aug_v4(self, results):
        dst_img = results['img']
        dst_mask = results['gt_semantic_seg']
        im_h, im_w = dst_img.shape[0:2]

        policy = random.choice(('rand_copy_move', 'rand_inpaint'))

        if policy == 'rand_copy_move':
            if random.random() > 0.5:
                x1, y1, x2, y2 = self._random_bbox(im_h, im_w)
                w, h = x2 - x1, y2 - y1

                # copy_move
                r_y = random.randint(0, im_h - h)
                r_x = random.randint(0, im_w - w)
                dst_img[r_y:r_y + h, r_x:r_x + w] = dst_img[y1:y2, x1:x2]
                dst_mask[r_y:r_y + h, r_x:r_x + w] = 1
            else:
                min_ = min(im_h, im_w)
                rand_radius1 = random.randint(int(min_*0.05), int(min_*0.2))
                rand_radius2 = random.randint(int(min_*0.05), int(min_*0.2))
                src_center_x = random.randint(rand_radius1+1, int(min_-rand_radius1-1))
                src_center_y = random.randint(rand_radius2+1, int(min_-rand_radius2-1))
                dst_center_x = random.randint(rand_radius1+1, int(min_-rand_radius1-1))
                dst_center_y = random.randint(rand_radius2+1, int(min_-rand_radius2-1))
                src_bool = self.generate_mask(im_h, im_w, rand_radius1, rand_radius2,
                                            src_center_x, src_center_y)
                dst_bool = self.generate_mask(im_h, im_w, rand_radius1, rand_radius2,
                                            dst_center_x, dst_center_y)
                dst_img[dst_bool] = dst_img[src_bool]
                dst_mask[dst_bool] = 1
        elif policy == 'rand_inpaint':
            x1, y1, x2, y2 = self._random_bbox(im_h, im_w)
            w, h = x2 - x1, y2 - y1

            y_min = max(y1 - 10, 0)
            x_min = max(x1 - 10, 0)
            y_max = min(y2 + 10, im_h)
            x_max = min(x2 + 10, im_w)
            h_min = max(0, y1 - y_min)
            w_min = max(0, x1 - x_min)
            # 使用cv2.inpaint方法，提高真实度，作为可选
            mask = np.zeros((y_max - y_min, x_max - x_min)).astype(np.uint8)
            mask[h_min:h_min+h, w_min:w_min+w] = 1
            inpainted_block = cv2.inpaint(dst_img[y_min:y_max, x_min:x_max], mask, 7, cv2.INPAINT_NS)
            dst_img[y1:y2, x1:x2] = inpainted_block[h_min:h+h_min, w_min:w+w_min]
            dst_mask[y1:y2, x1:x2] = 1

        results['img'] = dst_img
        results['gt_semantic_seg'] = dst_mask 
        if 'edge_gt_semantic_seg' in results:
            # generating edge mask
            results['edge_gt_semantic_seg'] = self.gen_edge(dst_mask)
            if 'edge_gt_semantic_seg' not in results['seg_fields']:
                results['seg_fields'].append('edge_gt_semantic_seg')
        if 'high_freq_img' in results:
            results['high_freq_img'] = gen_high_freq(dst_img, self.pipeline_cfg[0][0]['high_pass_filter_size'])

        # TODO: set to specific forgery class
        if 'img_label' in results:
            results['img_label'] = 1

        return results


@DATASETS.register_module()
class MaskSegDatasetv4(MaskSegDataset):
    def __init__(self,
                 pipeline,
                 data_root,
                 ann_path,
                 edge_mask_dir=None,
                 test_mode=False,
                 ignore_index=None,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None,
                 gt_seg_map_loader_cfg=None,
                 simulate_p=0.0,
                 dataset_name=''):
        self.pipeline_cfg = pipeline
        self.pre_pipelines = None
        if mmcv.is_list_of(pipeline, list):
            self.pre_pipelines = Compose(pipeline[0])
            self.post_pipelines = Compose(pipeline[1])
        else:
            self.post_pipelines = Compose(pipeline)

        self.data_root = data_root
        self.ann_path = ann_path
        self.edge_mask_dir = edge_mask_dir
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)
        self.gt_seg_map_loader = LoadAnnotations(binary=True
        ) if gt_seg_map_loader_cfg is None else LoadAnnotations(
            **gt_seg_map_loader_cfg)
        self.simulate_p = simulate_p
        self.dataset_name = dataset_name

        if test_mode:
            assert self.CLASSES is not None, \
                '`cls.CLASSES` or `classes` should be specified when testing'

        # load annotations
        self.img_infos = self.load_annotations()

    def load_annotations(self):
        img_infos = []

        anno_path = osp.join(self.data_root, self.ann_path)
        with open(anno_path) as f:
            lines = [line.strip() for line in f.readlines()]
    
        for line in lines:
            try:
                img_path, gt_path_= line.split(' ')
            except Exception as e:
                print(line)
                raise e

            img_path = osp.join(self.data_root, img_path)
            gt_path = osp.join(self.data_root, gt_path_)

            if not osp.isfile(img_path):
                continue
            if gt_path_ != 'None' and not osp.isfile(gt_path):
                continue

            img_info = dict(filename=img_path)
            img_info['ann'] = dict(seg_map=gt_path)
            # img_info['ann']['img_label'] = int(label)
            if self.edge_mask_dir:
                img_info['ann']['edge_mask'] = osp.join(self.edge_mask_dir, osp.basename(gt_path))
            img_infos.append(img_info)                

        # img_infos = sorted(img_infos, key=lambda x: x['filename'])

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = None
        results['seg_prefix'] = None

    def pre_eval(self, preds, indices, thresh=None):
        """Collect eval result from each iteration.

        Args:
            preds (list[torch.Tensor] | torch.Tensor): the segmentation logit
                after argmax, shape (N, H, W).
            indices (list[int] | int): the prediction related ground truth
                indices.

        Returns:
            list[torch.Tensor]: (area_intersect, area_union, area_prediction,
                area_ground_truth).
        """
        # In order to compat with batch inference
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]

        pre_eval_results = []
        for pred, index in zip(preds, indices):
            if not isinstance(pred, tuple):
                seg_pred = pred
                cls_pred = None
            else:
                cls_pred, seg_pred = pred
                # 1 is positive
                if seg_pred.shape[1] > 1:
                    seg_pred = seg_pred[:, 1, ...]
            # seg_map = self.get_gt_seg_map_by_idx(index).astype(pred.dtype)
            annos = self.get_gt_seg_map_by_idx(index)
            seg_gt = annos.get('gt_semantic_seg', None)
            if seg_gt is not None:
                seg_gt = seg_gt.astype(seg_pred.dtype)


            # if img_gt > 0 and seg_gt.max() > 0:

                # if seg_gt.max() == 0.:
                    # print(self.img_infos[index]['ann']['seg_map'])

            if thresh is None:
                seg_pred_ = seg_pred.flatten()
                seg_gt_ = seg_gt.astype(np.int).flatten()

                tpr, fpr, thresholds = metrics.roc_curve(seg_gt_, seg_pred_, pos_label=1)
                max_index = (tpr-fpr).argmax()
                thresh = thresholds[max_index]

            pred_l = (seg_pred > thresh).astype(np.float32)

            iou = intersect_and_union(pred_l, 
                                        seg_gt, 
                                        len(self.CLASSES),
                                        self.ignore_index, 
                                        self.label_map,
                                        self.reduce_zero_label)
            try:
                auc = calculate_auc(seg_pred.flatten(), seg_gt.flatten())
            except ValueError as e:
                auc = None


            score = np.max(seg_pred) if cls_pred is None else cls_pred

            pre_eval_results.append({'iou': iou, 'auc': auc, 'score': score})

        return pre_eval_results
    
    def multi_rand_aug(self, results):
        mask = results['gt_semantic_seg']

        if (mask==1).any() or random.random() > self.simulate_p:
            return results

        results = self.rand_aug_v4(results)

        return results

    @staticmethod
    def gen_edge(gt, epochs=12, kernel_size=8):
        gt = copy.deepcopy(gt)
        gt[gt==1] = 255

        kernel = np.ones((kernel_size, kernel_size), np.uint8) 
        gt = cv2.dilate(gt, kernel)

        edge_mask = np.zeros(gt.shape)
        for _ in range(epochs):
            # 形态学：边缘检测
            _,Thr_img = cv2.threshold(gt,128,255,cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
            gradient = cv2.morphologyEx(Thr_img, cv2.MORPH_GRADIENT, kernel)
            edge_mask[gradient==255] = 255
            gt[gradient==255] = 0
        edge_mask[edge_mask==255] = 1

        return edge_mask

    @staticmethod
    def generate_mask(img_height,img_width,radius1, radius2, center_x, center_y):
        y,x = np.ogrid[0:img_height,0:img_width]
        mask = ((x-center_x)/radius1)**2+((y-center_y)/radius2)**2<=1
        return mask

    def rand_aug_v4(self, results):
        dst_img = results['img']
        dst_mask = results['gt_semantic_seg']
        im_h, im_w = dst_img.shape[0:2]

        policy = random.choice(('rand_copy_move', 'rand_inpaint'))

        if policy == 'rand_copy_move':
            if random.random() > 0.5:
                x1, y1, x2, y2 = self._random_bbox(im_h, im_w)
                w, h = x2 - x1, y2 - y1

                # copy_move
                r_y = random.randint(0, im_h - h)
                r_x = random.randint(0, im_w - w)
                dst_img[r_y:r_y + h, r_x:r_x + w] = dst_img[y1:y2, x1:x2]
                dst_mask[r_y:r_y + h, r_x:r_x + w] = 1
            else:
                min_ = min(im_h, im_w)
                rand_radius1 = random.randint(int(min_*0.05), int(min_*0.2))
                rand_radius2 = random.randint(int(min_*0.05), int(min_*0.2))
                src_center_x = random.randint(rand_radius1+1, int(min_-rand_radius1-1))
                src_center_y = random.randint(rand_radius2+1, int(min_-rand_radius2-1))
                dst_center_x = random.randint(rand_radius1+1, int(min_-rand_radius1-1))
                dst_center_y = random.randint(rand_radius2+1, int(min_-rand_radius2-1))
                src_bool = self.generate_mask(im_h, im_w, rand_radius1, rand_radius2,
                                            src_center_x, src_center_y)
                dst_bool = self.generate_mask(im_h, im_w, rand_radius1, rand_radius2,
                                            dst_center_x, dst_center_y)
                dst_img[dst_bool] = dst_img[src_bool]
                dst_mask[dst_bool] = 1
        elif policy == 'rand_inpaint':
            x1, y1, x2, y2 = self._random_bbox(im_h, im_w)
            w, h = x2 - x1, y2 - y1

            y_min = max(y1 - 10, 0)
            x_min = max(x1 - 10, 0)
            y_max = min(y2 + 10, im_h)
            x_max = min(x2 + 10, im_w)
            h_min = max(0, y1 - y_min)
            w_min = max(0, x1 - x_min)
            # 使用cv2.inpaint方法，提高真实度，作为可选
            mask = np.zeros((y_max - y_min, x_max - x_min)).astype(np.uint8)
            mask[h_min:h_min+h, w_min:w_min+w] = 1
            inpainted_block = cv2.inpaint(dst_img[y_min:y_max, x_min:x_max], mask, 7, cv2.INPAINT_NS)
            dst_img[y1:y2, x1:x2] = inpainted_block[h_min:h+h_min, w_min:w+w_min]
            dst_mask[y1:y2, x1:x2] = 1

        results['img'] = dst_img
        results['gt_semantic_seg'] = dst_mask 
        if 'edge_gt_semantic_seg' in results:
            # generating edge mask
            results['edge_gt_semantic_seg'] = self.gen_edge(dst_mask)
            if 'edge_gt_semantic_seg' not in results['seg_fields']:
                results['seg_fields'].append('edge_gt_semantic_seg')
        if 'high_freq_img' in results:
            results['high_freq_img'] = gen_high_freq(dst_img, self.pipeline_cfg[0][0]['high_pass_filter_size'])

        # TODO: set to specific forgery class
        if 'img_label' in results:
            results['img_label'] = 1

        return results


    

