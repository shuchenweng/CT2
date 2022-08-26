import os
import torch
import torch.distributed


"""
GPU wrappers
"""

use_gpu = False
gpu_id = 0
device = None

distributed = False
dist_rank = 0
world_size = 1


def set_gpu_mode(mode, local_rank):
    global use_gpu
    global device
    global gpu_id
    global distributed
    global dist_rank
    global world_size
    gpu_id = 0 #int(os.environ.get('local_ranks')) #int(os.environ.get('LOCAL_RANK'))
    dist_rank = 0
    world_size = 1 #len(os.environ.get('CUDA_VISIBLE_DEVICES').split(',')) #len(os.environ.get('CUDA_VISIBLE_DEVICES').split(','))
    distributed = world_size > 1
    use_gpu = mode
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}" if use_gpu else "cpu")
    torch.backends.cudnn.benchmark = True
