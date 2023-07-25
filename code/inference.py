import cv2
cv2.setNumThreads(0)
import sys
import os
import os.path as osp
import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data
import argparse
import torchvision.transforms as transforms
import mmcv
from mmcv.utils import Config
from image_forgery_detection.datasets import build_dataset, build_dataloader
from image_forgery_detection.models import build_detector, build_segmentor
from image_forgery_detection.utils.evaluate import intersect_and_union, eval_metrics, pre_eval_to_metrics


transform_pil = transforms.Compose([
    transforms.ToPILImage(),
])


def get_opt():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument("--path", type=str, help="the model cfg and pth path")
    parser.add_argument("--test_dir", type=str, help="Path to the image list")
    parser.add_argument("--save_dir", type=str, default="./images")

    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    cudnn.benchmark = True

    opt = get_opt()
    print("in the head of inference:", opt)

    model_path = osp.join(opt.path, 'latest.pth')
    cfg_path = [osp.join(opt.path, f) for f in os.listdir(opt.path) if f.endswith('.py')][0]
    cfg = Config.fromfile(cfg_path)
    

    print(opt.test_dir)
    cfg.data.val[0].data_root = opt.test_dir 

    # cfg.data.val[0].test_mode = True
    test_dataset = build_dataset(cfg.data.val[0])
    test_dataloader = build_dataloader(
        test_dataset,
        samples_per_gpu=1,
        workers_per_gpu=2,
        dist=False,
        shuffle=False)

    model = build_segmentor(cfg.model)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')['state_dict']
        model.load_state_dict(checkpoint, strict=True)
        print(f"load {model_path} finish")
    else:
        print(f"{model_path} not exist")
        sys.exit()
    model.cuda()
    model.eval()

    os.makedirs(opt.save_dir, exist_ok=True)

    prog_bar = mmcv.ProgressBar(len(test_dataset))


    for ix, inputs in enumerate(test_dataloader):
        print(f"ttttest img idx: {ix}")
        # img is instance of DataContainer
        img = inputs['img'].data.cuda()
        img_meta = inputs['img_metas'].data[0]
        img_name = img_meta[0]['filename']

        with torch.no_grad():
            seg = model(img, img_meta, return_loss=False, rescale=False, argmax=False)[0]

        seg = seg.argmax(axis=0)
        seg = np.array(transform_pil(torch.from_numpy(seg.astype(np.uint8)))).astype(np.uint8)

        seg[seg == 1] = 255
        save_seg_path = os.path.join(opt.save_dir, os.path.split(img_name)[-1].split('.')[0] + '.jpg')
        cv2.imwrite(save_seg_path, seg)

        prog_bar.update()
