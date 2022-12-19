import torch
import torch.nn as nn


def get_lr_schedule(lr_schedule_type):
    """Return a learning rate scheduler function."""
    if lr_schedule_type == "constant":
        return lambda t: 1
    elif lr_schedule_type == "inverse":  # epoch wise
        return lambda t: 1 / (0.05 * t + 1)
    elif lr_schedule_type == "inverse_sqrt":  # for adam, epoch wise
        return lambda t: 1 / (t + 1) ** 0.5


def split_decay_params(model):
    """Split model parameters into groups for weight decay."""
    decay = set()
    no_decay = set()
    whitelist = (nn.Linear, nn.Conv2d)
    blacklist = (nn.BatchNorm2d,)
    for mn, m in model.named_modules():
        for pn, _ in m.named_parameters():
            fpn = "%s.%s" % (mn, pn) if mn else pn
            if "bias" in pn:
                no_decay.add(fpn)
            elif "weight" in pn and isinstance(m, whitelist):
                decay.add(fpn)
            elif "weight" in pn and isinstance(m, blacklist):
                no_decay.add(fpn)
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0
    assert len(param_dict.keys() - union_params) == 0
    return sorted(decay), sorted(no_decay)


@torch.no_grad()
def grad_norm(model, groups):
    """Calculate the norm of the gradients (per layer and total) for a model."""
    grads = []
    for pn, p in model.named_parameters():
        if p.grad is not None and pn in groups:
            grads.append(p.grad.data.flatten().pow(2).sum())
    return torch.stack(grads).sum().sqrt()


@torch.no_grad()
def weight_norm(model, groups):
    """Calculate the norm of the weights (per layer and total) for a model."""
    weights = []
    for pn, p in model.named_parameters():
        if pn in groups:
            weights.append(p.data.flatten().pow(2).sum())
    return torch.stack(weights).sum().sqrt()
