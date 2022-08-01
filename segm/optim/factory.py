from timm import scheduler
# from timm import optim
from torch import optim as optim

from segm.optim.scheduler import PolynomialLR
import torch.nn as nn


def create_scheduler(opt_args, optimizer):
    if opt_args.sched == "polynomial":
        lr_scheduler = PolynomialLR(
            optimizer,
            opt_args.poly_step_size,
            opt_args.iter_warmup,
            opt_args.iter_max,
            opt_args.poly_power,
            opt_args.min_lr,
        )
    else:
        lr_scheduler, _ = scheduler.create_scheduler(opt_args, optimizer)
    return lr_scheduler

def get_parameter_groups(model, weight_decay=0):
    parameter_list = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        parameter_list.append(param)
    return parameter_list


def create_optimizer(args, model, filter_bias_and_bn=True):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay        # 0
    if filter_bias_and_bn:
        print('finetune last 6 blocks in vit')
        parameters = get_parameter_groups(model, weight_decay)
    else:
        parameters = model.parameters()
        print('finetune all vit blocks..')
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        optimizer = optim.SGD(parameters, lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=weight_decay)
    else:
        assert False and "Invalid optimizer"
        raise ValueError
    return optimizer


