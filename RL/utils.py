import torch

def to_gpu_if_available(x):
    if torch.cuda.is_available():
        x = x.to('cuda')
    return x