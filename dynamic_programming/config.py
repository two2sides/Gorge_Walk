#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project :gorge_walk
@File    :config.py
@Author  :kaiwu
@Date    :2022/12/15 22:50

"""


# Configuration of dimensions
# 关于维度的配置
class Config:

    STATE_SIZE = 64 * 64
    ACTION_SIZE = 4
    GAMMA = 0.9
    THETA = 1e-3
    EPISODES = 100

    # dimensionality of the sample
    # 样本维度
    SAMPLE_DIM = 214

    # Dimension of movement action direction
    # 移动动作方向的维度
    OBSERVATION_SHAPE = 214

    # The following configurations can be ignored
    # 以下是可以忽略的配置
    LEGAL_ACTION_SHAPE = 0
    SUB_ACTION_MASK_SHAPE = 0
    LSTM_HIDDEN_SHAPE = 0
    LSTM_CELL_SHAPE = 0

    DIM_OF_ACTION = 4
