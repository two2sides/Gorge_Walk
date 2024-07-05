#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project :gorge_walk
@File    :train_workflow.py
@Author  :kaiwu
@Date    :2022/12/15 22:50

"""

from kaiwu_agent.utils.common_func import Frame
from kaiwu_agent.utils.common_func import attached
from monte_carlo.feature.definition import (
    observation_process,
    action_process,
    sample_process,
    reward_shaping,
)
import time
import math
import os


@attached
def workflow(envs, agents, logger=None, monitor=None):
    env, agent = envs[0], agents[0]
    EPISODES = 1000

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
    last_report_monitor_time = time.time()

    logger.info("Start Training...")
    start_t = time.time()

    total_rew, win_cnt = (
        0,
        0,
    )
    for episode in range(EPISODES):
        # User-defined environment launch configuration
        # 用户自定义的环境启动配置
        usr_conf = {
            "diy": {
                "start": [29, 9],
                "end": [11, 55],
                "treasure_id": [],
                # "treasure_num": 0,
                # "treasure_random": 0,
                # "max_step": 2000,
            }
        }

        # Reset the environment and obtain the initial state
        # 重置环境, 并获取初始状态
        obs = env.reset(usr_conf=usr_conf)

        # Disaster recovery
        # 容灾
        if obs is None:
            continue

        # First frame processing
        # 首帧处理
        obs_data = observation_process(obs)

        # Adjusting exploration rate based on win rate, with lower win rates leading to higher exploration and higher win rates leading to lower exploration
        # 根据胜率来调节探索率, 胜率越高探索越低, 胜率越低探索越高
        agent.epsilon = max(0.1, math.exp(-0.5 / (1 - win_cnt / (episode + 1))))

        act_data = agent.predict(list_obs_data=[obs_data])[0]
        act = action_process(act_data)
        init_frame = Frame(state=obs_data.feature, action=act, reward=0)

        # Task loop
        # 任务循环
        done, state_action_reward = False, [init_frame]
        while not done:
            # Interacting with the environment, performing actions, and obtaining the next state
            # 与环境交互, 执行动作, 获取下一步的状态
            frame_no, obs, score, terminated, truncated, env_info = env.step(act)
            if obs is None:
                break

            # Feature processing
            # 特征处理
            obs_data = observation_process(obs)

            # Compute reward
            # 计算 reward
            reward = reward_shaping(frame_no, score, terminated, truncated, obs)
            total_rew += reward

            # Agent performs inference to obtain the predicted action for the next frame
            # Agent 进行推理, 获取下一帧的预测动作
            act_data = agent.predict(list_obs_data=[obs_data])[0]

            # Unpacking ActData into actions
            # ActData 解包成动作
            act = action_process(act_data)

            # Determine over and update the win count
            # 判断结束, 并更新胜利次数
            done = terminated or truncated
            if terminated:
                win_cnt += 1
                # Processing the last frame
                # 处理最后一帧
                act = None

            # Updating data and generating frames for training
            # 数据更新, 生成训练需要的 frame
            frame = Frame(state=obs_data.feature, action=act, reward=reward)
            state_action_reward.append(frame)

        # Sample processing
        # 样本处理
        state_action_reward = sample_process(state_action_reward)

        # train
        # 训练
        agent.learn(state_action_reward)

        # Reporting training progress
        # 上报训练进度
        now = time.time()
        if now - last_report_monitor_time > 60:
            logger.info(f"Episode: {episode + 1}, Total Reward: {total_rew}")
            logger.info(f"Training Win Rate: {win_cnt / (episode + 1)}")
            monitor_data["reward"] = total_rew
            if monitor:
                monitor.put_data({os.getpid(): monitor_data})

            total_rew = 0
            last_report_monitor_time = now

        # The model has converged, training is complete, and reporting monitoring metric
        # 模型收敛, 结束训练, 上报监控指标
        if win_cnt / (episode + 1) > 0.9 and episode > 100:
            logger.info(f"Training Converged at Episode: {episode + 1}")
            monitor_data["reward"] = total_rew
            if monitor:
                monitor.put_data({os.getpid(): monitor_data})
            break

    end_t = time.time()
    logger.info(f"Training Time for {episode + 1} episodes: {end_t - start_t} s")
    agent.episodes = episode + 1

    # model saving
    # 保存模型
    agent.save_model()

    return