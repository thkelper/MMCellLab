export CUDA_VISIBLE_DEVICES=0
export PORT=62323
# SEED=1234
SEED=4567
GPUS_NUM=`GPUS=${CUDA_VISIBLE_DEVICES//,/}; echo ${#GPUS}`

PORT=$PORT CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES bash \
launch_train.sh /mnt/d/ycp/MMCellLab/code/MMColExp/config/unet_month3.py ${GPUS_NUM} --seed $SEED \
--work-dir  records/nested_unet_month3_seed_${SEED}

# PORT=$PORT CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES bash launch_train.sh ./image_forgery_detection/config/transforensics_guoshoucai_auto_forge_with_ps.py 4 --seed $SEED --work-dir  records/guoshoucai_auto_forge_with_ps_transforensics_baseline_seed_${SEED}

# PORT=$PORT CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES bash launch_train.sh ./image_forgery_detection/config/transforensics_multi_dataset_merge.py 4 --seed $SEED --work-dir  records/multi_dataset_merge_transforensics_baseline_seed_${SEED}

