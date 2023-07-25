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

def read_txt(fpath):
    img_list = list()
    with open(fpath, "r") as f:
        lines = f.readlines()
        for curline in lines:
            curline = curline.strip()
            img_list.append(curline)
    return img_list

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def rename_files(path):
    for root, dirs, files in os.walk(path):
        for dir_name in dirs:
            new_name = dir_name.replace(" ", "_")
            new_name = new_name.replace("+", "_")
            os.rename(osp.join(root, dir_name), osp.join(root, new_name))


if __name__ == "__main__":
    # dir_path = "/mnt/d/ycp/pku/unet++/input/asthma/0716_after_stretch"
    dir_path = "/mnt/d/ycp/pku/unet++/input/asthma/0716_dir_add_drug"

    rename_files(dir_path)