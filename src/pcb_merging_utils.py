import torch
import logging
import numpy as np
import copy
import csv
from functools import partial



def normalize(x, dim=0):
    min_values, _ = torch.min(x, dim=dim, keepdim=True)
    max_values, _ = torch.max(x, dim=dim, keepdim=True)
    y = (x - min_values) / (max_values - min_values)
    return y

def clamp(x, min_ratio=0, max_ratio=0):
    if len(x.size())==1:
        d = x.size(0)
        sorted_x, _ = torch.sort(x)
        min=sorted_x[int(d * min_ratio)]
        max=sorted_x[int(d * (1-max_ratio)-1)]
    else:
        d = x.size(1)
        sorted_x, _ = torch.sort(x, dim=1)
        min=sorted_x[:, int(d * min_ratio)].unsqueeze(1)
        max=sorted_x[:, int(d * (1-max_ratio)-1)].unsqueeze(1)
    clamped_x= torch.clamp(x, min, max)
    return clamped_x

def act(x):
    y = torch.tanh(x)  # torch.relu(x)
    return y

def PCB_merge(flat_task_checks, pcb_ratio=0.1):
    all_checks = flat_task_checks.clone()
    n, d = all_checks.shape   
    all_checks_abs = clamp(torch.abs(all_checks), min_ratio=0.0001, max_ratio=0.0001)
    clamped_all_checks = torch.sign(all_checks)*all_checks_abs
    self_pcb = normalize(all_checks_abs, 1)**2
    self_pcb_act = torch.exp(n*self_pcb)
    cross_pcb = all_checks * torch.sum(all_checks, dim=0)
    cross_pcb_act = act(cross_pcb)
    task_pcb = self_pcb_act * cross_pcb_act

    scale = normalize(clamp(task_pcb, 1-pcb_ratio, 0), dim=1)
    tvs = clamped_all_checks
    merged_tv = torch.sum(tvs * scale, dim=0) / torch.clamp(torch.sum(scale, dim=0), min=1e-12)
    return merged_tv, clamped_all_checks, scale

