'''
Author       : Thyssen Wen
Date         : 2022-05-09 14:54:32
LastEditors  : Thyssen Wen
LastEditTime : 2022-07-16 09:55:47
Description  : collect env info ref:https://github.com/open-mmlab/mmaction2/blob/master/mmaction/utils/collect_env.py
FilePath     : /ETESVS/utils/collect_env.py
'''
# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import collect_env as collect_basic_env
from mmcv.utils import get_git_hash
import torch 

def load_heads(model, weights):
    action_path = weights.replace('_latest_best', '_best_action')
    branch_path = weights.replace('_latest_best', '_best_branch')
    
    branch_weight = torch.load(branch_path)
    branch_weight = branch_weight['model_state_dict']
    branch_head_weights = {}
    for key, param in branch_weight.items():
        if 'branch_head' in key:
            branch_head_weights[key.replace('branch_head.', '')]   = param
    model.branch_head.load_state_dict(branch_head_weights)
    
    action_weight = torch.load(action_path)
    action_weight = action_weight['model_state_dict']
    action_head_weights = {}
    for key, param in action_weight.items():
        if 'action_head' in key:
            action_head_weights[key.replace('action_head.', '')]   = param
    model.action_head.load_state_dict(action_head_weights)

    return model

def collect_env():
    env_info = collect_basic_env()
    env_info['SVTAS'] = (get_git_hash(digits=7))
    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print(f'{name}: {val}')