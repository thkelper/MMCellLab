import os
import torch
import pynvml
import time

GPU_NUM = 8
# GPU_NUM = 4
PERCENT= 0.01
mu_gpus = []

tensors = []
pynvml.nvmlInit()
while True:
    if len(mu_gpus) < GPU_NUM:
        # for i in range(0, 8):
        # for i in range(4, 8):
        for i in range(0, GPU_NUM):
            if i in mu_gpus:
                continue
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_mem_percent = meminfo.used/meminfo.total
            print('GPU utilities: ',gpu_mem_percent)
            if gpu_mem_percent < PERCENT:
                #a = torch.zeros([100, 100, 100, 100, int(25*MEM)]).to('cuda:{}'.format(i))
                #tensors.append(a)
                mu_gpus.append(i)

        print(mu_gpus)
         
        if len(mu_gpus) < GPU_NUM:
            time.sleep(2*60)
    else:     
        break
