#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project :gorge_walk
@File    :train_workflow.py
@Author  :kaiwu
@Date    :2022/12/15 22:50

"""

from kaiwu_agent.utils.common_func import attached
import time
import os


@attached
def workflow(envs, agents, logger=None, monitor=None):
    env, agent = envs[0], agents[0]

    # Initializing monitoring data
    # 监控数据初始化
    monitor_data = {
        "reward": 0,
        "diy_1": 0,
        "diy_2": 0,
        "diy_3": 0,
        "diy_4": 0,
        "diy_5": 0,
    }

    logger.info("Start Training...")
    start_t = time.time()

    # Setting the state transition function
    # 设置状态转移函数
    from kaiwu_agent.gorge_walk.utils import get_F

    env.F = get_F("conf/map_data/F_level_1.json")

    agent.learn(env.F)

    logger.info(f"Training time cost: {time.time() - start_t} s")

    # Reporting training progress
    # 上报训练进度
    monitor_data["reward"] = 0
    if monitor:
        monitor.put_data({os.getpid(): monitor_data})

    # model saving
    # 保存模型
    agent.save_model()

    return
