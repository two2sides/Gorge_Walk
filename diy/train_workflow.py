#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project :gorge_walk
@File    :train_workflow.py
@Author  :kaiwu
@Date    :2022/12/15 22:50

"""

from diy.feature.definition import (
    observation_process,
    action_process,
    sample_process,
    reward_shaping,
)
from kaiwu_agent.utils.common_func import Frame
from kaiwu_agent.utils.common_func import attached
import time
import math
import os


@attached
def workflow(envs, agents, logger=None, monitor=None):
    """
    Users can define their own training workflows here
    用户可以在此处自行定义训练工作流
    """

    env, agent = envs[0], agents[0]

    # model saving
    # 保存模型
    # agent.save_model()

    return
