#!/usr/bin/env bash

python ./ensembel_inference.py \
    --model_paths ../user_data/models/fpn ../user_data/models/deeplabv3 \
    --test_dir ../test_data \
    --save_dir ../prediction_result/images

#python ./ensembel_inference.py \
#    --model_paths ../user_data/models/fpn ../user_data/models/deeplabv3 \
#    --test_dir ../test_data/img \
#    --save_dir ../prediction_result/images \
#    --local_eval
