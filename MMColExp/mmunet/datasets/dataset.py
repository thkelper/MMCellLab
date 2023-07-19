import os
import os.path as osp
import cv2
import numpy as np
import torch
from collections import OrderedDict
import torch.utils.data
from .builder import DATASETS
import mmcv
from mmcv.utils import print_log
from ..utils.logger import get_root_logger
from .pipelines import Compose
import warnings
from ..utils.evaluate import (intersect_and_union, eval_metrics, pre_eval_to_metrics,
                            calculate_auc, calculate_img_score)
from prettytable import PrettyTable
from sklearn import metrics
            

@DATASETS.register_module()
class Dataset(torch.utils.data.Dataset):
    CLASSES = ["medium", "collagen"]
    PALETTE = [0, 1]
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))

        mask = []
        for i in range(self.num_classes):
            mask.append(cv2.imread(os.path.join(self.mask_dir, str(i),
                        img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        mask = np.dstack(mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)#这个包比较方便，能把mask也一并做掉
            img = augmented['image']#参考https://github.com/albumentations-team/albumentations
            mask = augmented['mask']
        
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)
        
        return img, mask, {'img_id': img_id}


@DATASETS.register_module()
class MMDataset(torch.utils.data.Dataset):
    CLASSES = ["medium", "collagen"]
    PALETTE = [0, 1]

    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix='.bmp',
                 ann_dir=None,
                 seg_map_suffix='.tif',
                 split=None,
                 data_root=None,
                 test_mode=True,
                 ignore_index=None,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None,
                 gt_set_map_loader_cfg=None,
                 simulate_p=0.5,
                 dataset_name='',
                 **unused_kwargs,
                 ) -> None:
        self.pre_pipelines = None
        if mmcv.is_list_of(pipeline, list):
            self.pre_pipelines = Compose(pipeline[0])
            self.post_pipelines = Compose(pipeline[1])
        else:
            self.post_pipelines = Compose(pipeline)

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
        self.CLASSES, self.PALETTE = self.get_classes_and_palette

        self.dataset_name = dataset_name

        if test_mode:
            assert self.CLASSES is not None, \
            '`cls.CLASSES` or `classes` should be specified when testing'

        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.split is None or osp.isabs(self.split)):
                    self.split = osp.join(self.data_root, self.split)


        self.img_infos = self.load_annotations(self.img_dir, self.img_suffix,
                                               self.ann_dir, self.seg_map_suffix,
                                                self.split)

        def __len__(self):
            return len(self.img_infos)
        
        def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split):
            
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
                for img in mmcv.scandir(img_dir,  img_suffix, recursive=True):
                    img_info = dict(filename=img)
                    if ann_dir is not None:
                        seg_map = img.replace(img_suffix, seg_map_suffix)
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
                img_infos = sorted(img_infos, key=lambda x: x['filename'])
            
            print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
            return img_infos       

    def get_ann_info(self, idx):
        return self.img_infos[idx]['ann']

    def get_img_info(self, idx):
        return self.img_infos[idx]
    
    def pre_pipeline(self, results):
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            ret = self.prepare_train_img(idx)
            return ret
            
    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info[idx]
        results = dict(img_info=img_info, ann_info=ann_info) 
        self.pre_pipeline(results)
        if self.pre_pipelines:
            results = self.pre_pipelines(results) 
            #results = self.multi_rand_aug(results)
        results = self.post_pipelines(results)

        return results


    def prepare_test_img(self, idx):
        try:
            img_info = self.img_infos[idx]
            results = dict(img_info=img_info)
            self.prepiline(results)
            if self.pre_pipelines:
                results = self.pre_pipelines(results)
            results = self.post_pipelines(results)
        except Exception as e:
            raise e
        return results


    def get_gt_seg_map_by_idx(self, index):
        ann_info = self.get_ann_info(index)
        results = dict(ann_info=ann_info)
        self.pre_pipeline(results)
        self.get_seg_map_loader(results)
        return results

    def get_gt_seg_maps(self, efficient_test=None):
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
















@DATASETS.register_module()
class DatasetV2(torch.utils.data.Dataset):
    def __init__(self, img_ids, ann_list,  transform=None, mode="train", data_root=None):
        
        self.img_ids = img_ids
        self.ann_list = ann_list
        self.transform = transform
        self.mode = mode
        self.data_root = data_root

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        # img_fpath, mask_fpath = self.ann_list[img_id].split(" ") 
         
        
        if self.mode == "train":
            img_fpath, mask_fpath = img_id.split(" ")
            if self.data_root:
                img_fpath = osp.join(self.data_root, img_fpath)
                mask_fpath = osp.join(self.data_root, mask_fpath)
                if not osp.exists(img_fpath):
                    print(img_fpath)
                if not osp.exists(mask_fpath):
                    print(mask_fpath)
            img = cv2.imread(img_fpath)
            # print(f"img.shape:{img.shape}")
            mask = cv2.imread(mask_fpath)
            # print(f"img.shape:{img.shape}")
        elif self.mode == "infer": 
            img_fpath = img_id.strip()
            # print(f"before:{img_fpath}")
            img_fpath = osp.join(self.data_root, img_fpath)
            # print(f"after:{img_fpath}")
            img = cv2.imread(img_fpath)
            mask = np.zeros_like(img)

        # mask = []
        # for i in range(self.num_classes):
        #     mask.append(cv2.imread(os.path.join(self.mask_dir, str(i),
        #                 img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        # mask = np.dstack(mask) 
        img = np.array(img)
        mask = np.array(mask)
        if self.transform is not None:
            # print(f"img.shape:{img.shape}   mask.shape:{mask.shape}")
            augmented = self.transform(image=img, mask=mask)#这个包比较方便，能把mask也一并做掉
            img = augmented['image']#参考https://github.com/albumentations-team/albumentations
            mask = augmented['mask']
        
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)
        mask = mask[0,:,:]
        mask = mask[np.newaxis, :, :]
        img_fname = "_".join(img_fpath.split('/')[-3:]) 
        return img, mask, {'img_id': img_fname, 'img_fpath': img_fpath} 