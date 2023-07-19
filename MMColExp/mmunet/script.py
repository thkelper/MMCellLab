import argparse
import os
import os.path as osp

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def generate_ann_txt(root_dir, ann_save_dir, img_ext=".bmp", mask_ext=".tif", mode="train"):
    all_ann_info = []
    for cur_root, dirs, files in os.walk(root_dir):
        for file in files:
            fpath = osp.join(cur_root, file)
            if img_ext in fpath and not "._" in fpath:
                mask_path = fpath.replace(img_ext, mask_ext)
                if osp.exists(mask_path):
                    all_ann_info.append(fpath + " " + mask_path + "\n")
                elif mode=="infer":
                    # all_ann_info.append(fpath + " " + mask_path + "\n")
                    all_ann_info.append(fpath + "\n") 
                else:
                    print(mask_path)
                    continue
    with open(osp.join(ann_save_dir, "ann_info.txt"), "w") as f:
        f.writelines(all_ann_info)

def read_txt(fpath):
    img_list = list()
    with open(fpath, "r") as f:
        lines = f.readlines()
        for curline in lines:
            curline = curline.strip()
            img_list.append(curline)
    return img_list

if __name__ == "__main__":
    # root_dir = "/mnt/disk1/data/mouse_det/month3"
    # root_dir = "/mnt/disk1/data/mouse_det/0329_11am_cell800k_col1.0"
    # root_dir = "/mnt/d/ycp/pku/unet++/input/anti_cancer_phase"
    root_dir = "/mnt/d/ycp/pku/unet++/input/anti_cancer_phase/0628_48well"
    ann_save_dir = root_dir
    generate_ann_txt(root_dir, ann_save_dir, mode="infer")
    # ann_file_path = osp.join(root_dir, "ann_info.txt")
    # test_list = read_txt(ann_file_path)
    