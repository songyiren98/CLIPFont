import torch
from copy import deepcopy


def decouple_shape_groups(shape_groups):
    shape_groups_decouple = []
    for group in shape_groups:
        for id in group.shape_ids.detach().cpu().numpy():
            group_decouple = deepcopy(group)
            group_decouple.shape_ids = torch.Tensor([id]).to(torch.int32)
            shape_groups_decouple.append(group_decouple)
    return shape_groups_decouple


def get_color_vars(shape_groups, use_blob: bool = True):
    color_vars = {}
    for group in shape_groups:
        if use_blob:
            group.fill_color.requires_grad = True
            color_vars[group.fill_color.data_ptr()] = group.fill_color
        else:
            group.stroke_color.requires_grad = True
            color_vars[group.stroke_color.data_ptr()] = group.fill_color
    color_vars = list(color_vars.values())
    return color_vars
