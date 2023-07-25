import cv2
cv2.setNumThreads(0)
import sys
import os
import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data
import argparse
import torchvision.transforms as transforms
import mmcv
from mmcv.utils import Config
from image_forgery_detection.datasets import build_dataset, build_dataloader
from image_forgery_detection.models import build_detector
from image_forgery_detection.utils.evaluate import intersect_and_union, eval_metrics, pre_eval_to_metrics


transform_pil = transforms.Compose([
    transforms.ToPILImage(),
])


def get_opt():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument("--model_paths", type=str, nargs='+')
    parser.add_argument("--test_dir", type=str, help="Path to the image list")
    parser.add_argument("--save_dir", type=str, default="./images")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--local_eval", action='store_true')

    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    cudnn.benchmark = True

    opt = get_opt()
    print("in the head of inference:", opt)

    model_paths = list(os.path.join(path, 'latest.pth') for path in opt.model_paths)
    configs = []
    for path in opt.model_paths:
        configs.append(list(os.path.join(path, f) for f in os.listdir(path) if f.endswith('.py'))[0])

    weights = opt.weights
    if weights is None:
        weights = [1/len(configs)] * len(configs)
    else:
        weights = [float(item) for item in weights.split(';')]

    cfg = Config.fromfile(configs[0])

    print(opt.test_dir)
    cfg.data.val.img_dir = opt.test_dir
    cfg.data.val.ann_dir = None
    cfg.data.val.data_root = None
    test_dataset = build_dataset(cfg.data.val)
    test_dataloader = build_dataloader(
        test_dataset,
        samples_per_gpu=1,
        workers_per_gpu=2,
        dist=False,
        shuffle=False)

    models = []
    for config, model_path in zip(configs, model_paths):
        cfg = Config.fromfile(config)
        cfg.model.base_model.pretrined_path = '../user_data/pretrained/pvt_v2_b3.pth'
        # load model
        model = build_detector(cfg.model)
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')['state_dict']
            model.load_state_dict(checkpoint, strict=True)
            print(f"load {model_path} finish")
        else:
            print(f"{model_path} not exist")
            sys.exit()
        model.cuda()
        model.eval()
        models.append(model)

    os.makedirs(opt.save_dir, exist_ok=True)

    prog_bar = mmcv.ProgressBar(len(test_dataset))

    if opt.local_eval:
        pre_eval_results = []

    for ix, inputs in enumerate(test_dataloader):
        # img is instance of DataContainer
        img = inputs['img'].data.cuda()
        img_meta = inputs['img_metas'].data[0]
        img_name = img_meta[0]['filename']

        segs = []
        with torch.no_grad():
            for model in models:
                seg = model(img, img_meta, return_loss=False, rescale=False, argmax=False)[0]
                segs.append(seg)

        seg = sum(seg * w for seg, w in zip(segs, weights))
        seg = seg.argmax(axis=0)
        seg = np.array(transform_pil(torch.from_numpy(seg.astype(np.uint8)))).astype(np.uint8)

        if opt.local_eval:
            mask_path = img_name.replace('img/', 'mask/').replace('jpg', 'png')
            mask = torch.from_numpy(mmcv.imread(mask_path, flag='unchanged', backend='pillow'))
            mask[mask < 127] = 0
            mask[mask >= 127] = 1
            pre_eval_results.append(intersect_and_union(seg, mask, 2, None, None))
        else:
            seg[seg == 1] = 255
            save_seg_path = os.path.join(opt.save_dir, os.path.split(img_name)[-1].split('.')[0] + '.png')
            cv2.imwrite(save_seg_path, seg)

        prog_bar.update()

    if opt.local_eval:
        test_dataloader.dataset.evaluate(pre_eval_results, metric='IoU+f1')
