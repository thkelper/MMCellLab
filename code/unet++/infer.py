import argparse
import os
import os.path as osp
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
# from albumentations.augmentations import transforms
import albumentations.augmentations as A
from albumentations.core.composition import Compose
# from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils import read_txt, AverageMeter
import archs
from dataset import DatasetV2
from metrics import iou_score
"""
需要指定参数：--name dsb2018_96_NestedUNet_woDS
"""
IMG_EXT = ".tif"
MASK_EXT = ".tif"
PRED_EXT = ".jpg"

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default="agarose_train_input_512",
                        help='model name')
    # parser.add_argument('--infer_ann_file', default="/mnt/d/ycp/pku/unet++/input/anti_cancer_phase/all_info.txt")
    parser.add_argument('--infer_ann_file', default="/mnt/d/ycp/data/agarose_stamp/agarose_0718_val/all_info.txt")
    parser.add_argument('--vis_feature', default=False)
    args = parser.parse_args()

    return args


def infer(args):

    with open('./models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'],
                                           vis_feature=args.vis_feature)
    # device = torch.device("cpu") 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)

    # Data loading code
    # img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    # img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    # _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)
    img_mask_list = read_txt(args.infer_ann_file)
    # img_ids = list(range(len(img_mask_list))) 

    model.load_state_dict(torch.load('./models/%s/model.pth' %
                                     config['name']))
    model.eval() 

    val_transform = Compose([
        A.Resize(config['input_h'], config['input_w']),
        A.Normalize(),
    ])

    val_dataset = DatasetV2( 
        img_ids=img_mask_list, 
        ann_list=img_mask_list,
        transform=val_transform,
        mode="infer",
        data_root=osp.dirname(args.infer_ann_file))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        # batch_size=1,
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)


    # for c in range(config['num_classes']):
    os.makedirs(os.path.join('outputs', config['name']), exist_ok=True)
    with torch.no_grad():
        print(f"len(val_loader):{val_loader}")
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            
            # print(f"input.shape：{input.shape}, target.shape:{target.shape}, meta:{meta}")
            input = input.to(device)
            target = target.to(device)
            
            # compute output
            if config['deep_supervision']:
                output = model(input)[-1]
            elif args.vis_feature:
                output, vis_features = model(input)
            elif not args.vis_feature:
                output = model(input)

            # iou = iou_score(output, target)
            # avg_meter.update(iou, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()
            
            for i in range(len(output)):
                print(meta["img_fpath"][i])

                # new_img_fname = meta['img_id'][i]
                # new_img_prefix, new_img_suffix = osp.splitext(new_img_fname)
                # new_img_fname = new_img_prefix + '.jpg'
                new_img_save_path = meta['img_fpath'][i].replace(IMG_EXT, PRED_EXT)
                
                pred_mask = (output[i] * 255).astype('uint8').transpose(1, 2, 0)
                ori_img_fpath = meta['img_fpath'][i]
                ori_img = cv2.imread(ori_img_fpath)
                ori_img_h, ori_img_w = ori_img.shape[:2]
                pred_mask = cv2.resize(pred_mask, (ori_img_w, ori_img_h), interpolation=cv2.INTER_LINEAR ) 
                
                cv2.imwrite(new_img_save_path, pred_mask)
                

                if args.vis_feature:
                    vis_features_dir = osp.join(osp.dirname(args.infer_ann_file), "vis_feature")
                    os.makedirs(vis_features_dir, exist_ok=True)
                    for fea_name, fea in vis_features.items():
                        if isinstance(fea, torch.Tensor):
                            fea = (fea.cpu().numpy()[i][0,:,:] * 255).astype('uint8') 
                            cv2.resize(fea, (ori_img_w, ori_img_h), interpolation=cv2.INTER_LINEAR ) 
                            fea_name += PRED_EXT
                            cv2.imwrite(osp.join(vis_features_dir, fea_name), fea) 
                print(f"{new_img_save_path} processed")


def plot_examples(datax, datay, model,num_examples=6):
    fig, ax = plt.subplots(nrows=num_examples, ncols=3, figsize=(18,4*num_examples))
    m = datax.shape[0]
    for row_num in range(num_examples):
        image_indx = np.random.randint(m)
        image_arr = model(datax[image_indx:image_indx+1]).squeeze(0).detach().cpu().numpy()
        ax[row_num][0].imshow(np.transpose(datax[image_indx].cpu().numpy(), (1,2,0))[:,:,0])
        ax[row_num][0].set_title("Orignal Image")
        ax[row_num][1].imshow(np.squeeze((image_arr > 0.40)[0,:,:].astype(int)))
        ax[row_num][1].set_title("Segmented Image localization")
        ax[row_num][2].imshow(np.transpose(datay[image_indx].cpu().numpy(), (1,2,0))[:,:,0])
        ax[row_num][2].set_title("Target image")
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    infer(args)
