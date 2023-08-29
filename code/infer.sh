#!/usr/bin/env bash

python ./inference.py \
    --path ./records/nested_unet_month3_seed_4567 \
    --test_dir /mnt/d/ycp/data/a549_col/month3 \
    --save_dir ./images \
    # --local_infer
# --test_dir /mnt/d/ycp/data/asthma/inference_test/0h \
#python ./ensembel_inference.py \
#    --model_paths ../user_data/models/fpn ../user_data/models/deeplabv3 \
#    --test_dir ../test_data/img \
#    --save_dir ../prediction_result/images \
#    --local_eval
