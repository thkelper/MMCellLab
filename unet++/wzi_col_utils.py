#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os 
import os.path as osp
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from glob import glob
import re
from tabulate import tabulate


# In[3]:




# In[4]:


def save_fname_txt(cfgs, src_dir, save_dir=None, save_fname="all_info.txt", with_mask=False, save=True, ret=False):
    assert osp.isdir(src_dir), "Must input the dir of images."
    
    all_img_info_list = list()
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.startswith("._"): continue
            if file.endswith(cfgs.IMG_EXT):
                img_abs_path = osp.join(root, file)
                mask_abs_path = img_abs_path.replace(cfgs.IMG_EXT,cfgs.MASK_EXT)
                img_path = osp.join(root.replace(src_dir, "."), file)
                mask_path = img_path.replace(cfgs.IMG_EXT, cfgs.MASK_EXT)
                if osp.exists(mask_abs_path) and with_mask:
                    all_img_info_list.append(img_path + " " + mask_path + "\n")
                else:
                    all_img_info_list.append(img_path + "\n")
    if not save_dir:
        txt_save_path = osp.join(src_dir, save_fname)
    else:
        txt_save_path = osp.join(save_dir, save_fname)
    if save:
        with open(txt_save_path, "w") as fp:
            fp.writelines(sorted(all_img_info_list))
    if ret:
        return sorted(all_img_info_list)


# In[5]:


def save_txt_with_medicines(cfgs, cur_work_dir, all_medicines, save_fname_EXT="all_info.txt", verbose=False):
    for medicine in all_medicines:
        medicine_list = sorted([item for item in save_fname_txt(cfgs, cur_work_dir, save_dir=cur_work_dir, save=False, ret=True) if re.search(medicine, item)])
        medicine_txt_fpath = osp.join(cur_work_dir, medicine + '_' + save_fname_EXT)
        with open(medicine_txt_fpath, "w") as fb:
            fb.writelines(medicine_list)
        if verbose:
            print(f'{medicine_txt_fpath}  saved') 


# In[6]:


def statistic_area(cfgs, all_info_path, save_dir, save=True, count_area=True):
#     if osp.isdir(all_info_path):
    all_area_info = list()
    all_area_dict = dict()
    with open(all_info_path) as fb:
        for line in fb.readlines():
            img_path = line.strip()
            img_path = osp.join(save_dir, img_path)
            mask_path = img_path.replace(cfgs.IMG_EXT, cfgs.MASK_EXT)
            
            pixel_counts = None
            if count_area:
                mask = cv2.imread(mask_path)
                thresh, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
                pixel_counts = np.count_nonzero(mask)
            new_img_fname = "_".join(mask_path.split("/")[-3:])
            time, medicine, img_fname = mask_path.split("/")[-3:]
            all_area_dict[new_img_fname] = dict(fname=img_fname, time=time, medicine=medicine, area=pixel_counts)
            all_area_info.append(new_img_fname + " " + str(pixel_counts) + "\n") 
    if save:
        with open(osp.join(save_dir, "all_area_info.txt"), "w") as fb:
            fb.writelines(all_area_info)
    return all_area_dict


# In[7]:


# work_dir = p_anti_cancer_dir; all_medicines = p_anti_medicines; save_fname_EXT="all_info.txt"
# work_dir = s_anti_cancer_dir; all_medicines = s_anti_medicines; save_fname_EXT="all_info.txt"
def rm_not_valid_imgs_plot(cfgs, work_dir, all_medicines=None, valid_times=None, save_fname_EXT="all_info.txt", verbose=False):

    n_rows = 1; n_cols = 6
# if True:
    figure,axes=plt.subplots(nrows=n_rows,ncols=n_cols,figsize=(20,4), sharex=True, sharey=True) 

    cur_area_dict = dict()
    medicine_mean_std_res = pd.DataFrame()
    for i, medicine in enumerate(all_medicines):

        cur_txt_path = osp.join(work_dir, medicine + "_" + save_fname_EXT)
        cur_area_dict = statistic_area(cfgs, cur_txt_path, work_dir, save=False)
        cur_df = pd.DataFrame(cur_area_dict).transpose()
        cur_valid_area_df = cur_df.groupby('fname').count()[cur_df.groupby('fname').count()['time']>=5]
        cur_valid_imgs = cur_valid_area_df.index.tolist()
        cur_valid_df = cur_df[cur_df['fname'].isin(cur_valid_imgs)]
        cur_valid_df = cur_valid_df.set_index('fname')
        groups = cur_valid_df.groupby("fname")
        fname2area = dict()
        for k, v in groups:

#             v = v.drop(columns="medicine", axis=1)
            fname = v.index.to_list()[0]
            time = v['time'].to_list()
            area = v['area'].to_list()
            fname2area[fname] = area

        if fname2area:
            ultra_df = pd.DataFrame(fname2area).transpose().set_axis(time, axis=1)
            rate_ultra_df = ultra_df.div(ultra_df['0h'],axis=0)
#             print(f"rate_ultra_df\n:{rate_ultra_df}")
#             rate_ultra_df.boxplot(ax=axes[i], column=time, xlabel=medicine)
            df_mean = rate_ultra_df.mean().to_frame()
            df_std = rate_ultra_df.std().to_frame()
            df_mean.plot(ax=axes[i], xlabel=medicine)
            
#             print(f"m:\n{tabulate(rate_ultra_df.mean().to_frame().transpose(), headers='keys')}")
#             print(f"s:\n{tabulate(rate_ultra_df.std().to_frame().transpose(), headers='keys')}")
            df_mean.columns = [f'{medicine}_mean']
            df_std.columns = [f'{medicine}_std']
            mean_std = pd.concat([df_mean, df_std], axis=1)
            medicine_mean_std_res = pd.concat([medicine_mean_std_res, mean_std], axis=1)
            if verbose:
                print(f"medicine:{medicine}")
                print(f"m:\n{df_mean}")
                print(f"s:\n{df_std}")
                print(f"mean_std:\n{mean_std}")
                print("-" * 50)
        else:
            print(f"the data of {medicine} ERROR!")
    assay_name = work_dir.split("/")[-1]
    medicine_mean_std_res.to_excel(osp.join(work_dir, assay_name + ".xlsx"))


# In[13]:




# In[16]:


# work_dir = p_anti_cancer_dir; all_medicines = p_anti_medicines; save_fname_EXT="all_info.txt"
# work_dir = s_anti_cancer_dir; all_medicines = s_anti_medicines; save_fname_EXT="all_info.txt"
def rm_not_valid_imgs_plot(cfgs, work_dir, all_medicines=None, save_fname_EXT="all_info.txt", valid_thresh=None, verbose=False):
# verbose = False
    n_cols = 4; n_rows = int(np.ceil(len(all_medicines) / n_cols)); 
# if True:
    figure,axes=plt.subplots(nrows=n_rows,ncols=n_cols,figsize=(20,10), squeeze=False, sharex=True, sharey=True) 

    cur_area_dict = dict()
    medicine_mean_std_res = pd.DataFrame()
    for i, medicine in enumerate(all_medicines):

        cur_txt_path = osp.join(work_dir, medicine + "_" + save_fname_EXT)
        cur_area_dict = statistic_area(cfgs, cur_txt_path, work_dir, save=False)
        cur_df = pd.DataFrame(cur_area_dict).transpose()
        
        cur_valid_area_df = cur_df.groupby('fname').count()[cur_df.groupby('fname').count()['time']>= valid_thresh]
        cur_valid_imgs = cur_valid_area_df.index.tolist()
        cur_valid_df = cur_df[cur_df['fname'].isin(cur_valid_imgs)]
        cur_valid_df = cur_valid_df.set_index('fname')
        groups = cur_valid_df.groupby("fname")
        fname2area = dict()
        for k, v in groups:
#             v = v.drop(columns="medicine", axis=1)
            fname = v.index.to_list()[0]
            time = v['time'].to_list()
            print(time)
            # time = sorted(time, key=lambda date: float(date[:-1]))
            time = sorted(time, key=lambda data: float(data[1:]))
            area = v['area'].to_list()
            fname2area[fname] = area

        if fname2area:
            ultra_df = pd.DataFrame(fname2area).transpose().set_axis(time, axis=1)
            rate_ultra_df = ultra_df.div(ultra_df[ultra_df.columns[0]],axis=0)
            rate_ultra_df = 1 - rate_ultra_df
#             print(f"rate_ultra_df\n:{rate_ultra_df}")
#             rate_ultra_df.boxplot(ax=axes[i], column=time, xlabel=medicine)
            df_mean = rate_ultra_df.mean().to_frame()
            df_std = rate_ultra_df.std().to_frame()
            df_mean.plot(ax=axes[i // n_cols][i % n_cols], title=medicine, ylabel="shrink_rate")
            
#             print(f"m:\n{tabulate(rate_ultra_df.mean().to_frame().transpose(), headers='keys')}")
#             print(f"s:\n{tabulate(rate_ultra_df.std().to_frame().transpose(), headers='keys')}")
            df_mean.columns = [f'{medicine}_mean']
            df_std.columns = [f'{medicine}_std']
            mean_std = pd.concat([df_mean, df_std], axis=1)
            medicine_mean_std_res = pd.concat([medicine_mean_std_res, mean_std], axis=1)
            if verbose:
                print(f"medicine:{medicine}")
                # print(f"m:\n{df_mean}")
                # print(f"s:\n{df_std}")
                print(f"mean_std:\n{mean_std}")
                print("-" * 50)
        else:
            print(f"the data of {medicine} ERROR!")
    
    assay_name = work_dir.split("/")[-1]
    plt.suptitle(assay_name)
    img_fname = osp.join(work_dir, assay_name + ".jpg")
    plt.savefig(img_fname)
    medicine_mean_std_res.to_excel(osp.join(work_dir, assay_name + ".xlsx"))


# In[ ]:




