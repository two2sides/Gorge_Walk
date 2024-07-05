#!/usr/bin/env python3
# -*- coding:utf-8 -*-


"""
@Project :gorge_walk
@File    :model.py
@Author  :kaiwu
@Date    :2022/11/15 20:57

"""

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from kaiwu_agent.utils.common_func import attached


class Model(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()

        # User-defined network
        # 用户自定义网络
