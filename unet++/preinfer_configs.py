import os.path as osp

# root_dir = "/mnt/d/ycp/pku/unet++/input/asthma/0716_dir_add_drug"
# root_dir = "/mnt/d/ycp/pku/unet++/input/asthma/0716_after_stretch"
root_dir = "/mnt/d/ycp/data/20230711_Mono_Image_time_series"
ann_save_dir = root_dir
IMG_EXT = ".tif"
MASK_EXT = ".jpg"
# medicines = ['1uM_Ach', '1uM_Ach_0.0001nM_TB', '1uM_Ach_0.001nM_TB', '1uM_Ach_0.01nM_TB', '1uM_Ach_0.1nM_TB',
            #  '1uM_Ach_1nM_TB', '1uM_Ach_10nM_TB', '1uM_Ach_100nM_TB', '1uM_Ach_1000nM_TB', 'Control']
medicines = ['1uM_Ach', '1uM_Ach_0.0001nM_TB', '1uM_Ach_0.001nM_TB', '1uM_Ach_0.01nM_TB', '1uM_Ach_0.1nM_TB',
             '1uM_Ach_1nM_TB', '1uM_Ach_10nM_TB', '1uM_Ach_100nM_TB', '1uM_Ach_1000nM_TB'] 
    
# medicines = ['t0', 't7', 't8']
valid_thresh = 2
start_idx = 0 
infer = False
# infer = True
end_idx = 3

# infer_configs 
name = "agarose_train_input_512"
infer_ann_file = osp.join(root_dir, "ann_info.txt")
vis_feature = False
verbose = True