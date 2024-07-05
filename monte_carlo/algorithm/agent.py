#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project :gorge_walk
@File    :agent.py
@Author  :kaiwu
@Date    :2022/12/15 22:50

"""

import numpy as np
import kaiwu_agent
from kaiwu_agent.agent.base_agent import (
    predict_wrapper,
    exploit_wrapper,
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
)
from kaiwu_agent.utils.common_func import attached
from monte_carlo.feature.definition import ActData
from kaiwu_agent.agent.base_agent import BaseAgent
from monte_carlo.config import Config


@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None) -> None:
        self.logger = logger

        # Initialize environment parameters
        # 参数初始化
        self.state_size = Config.STATE_SIZE
        self.action_size = Config.ACTION_SIZE

        self.gamma = Config.GAMMA
        self.epsilon = Config.EPSILON
        self.episodes = Config.EPISODES

        # Initialize the policy
        # 初始化策略
        self.policy = np.random.choice(self.action_size, self.state_size)
        self.Q = np.zeros([self.state_size, self.action_size])
        self.visit = np.zeros([self.state_size, self.action_size])

        super().__init__(agent_type, device, logger, monitor)

    @predict_wrapper
    def predict(self, list_obs_data):
        """
        The input is list_obs_data, and the output is list_act_data.
        """
        """
        输入是 list_obs_data, 输出是 list_act_data
        """
        state = list_obs_data[0].feature
        act = self._epsilon_greedy(state=state, epsilon=self.epsilon)

        return [ActData(act=act)]

    @exploit_wrapper
    def exploit(self, list_obs_data):
        state = list_obs_data[0].feature
        act = self.policy[state]

        return [ActData(act=act)]

    def _epsilon_greedy(self, state, epsilon=0.1):
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_size)
        else:
            return self.policy[state]

    @learn_wrapper
    def learn(self, list_sample_data):
        """
        Calculate the optimal policy using Monte Carlo Control - fist visit
            - list_sample is a list a sample: (state, action, reward)
        Return is calculated using the following formula:
            - G = R(t+1) + gamma * R(t+2) + ... + gamma^(T-t-1) * R(T)
        """
        """
        使用蒙特卡洛控制 - 首次访问来计算最优策略
        - list_sample 是一个样本列表：(状态, 动作, 奖励)
        使用以下公式计算返回值：
        - G = R(t+1) + gamma * R(t+2) + ... + gamma^(T-t-1) * R(T)
        """
        G, state_action_return = 0, []

        # Calculate the return for each state-action pair
        # 计算每个状态-动作对的回报
        for sample in reversed(list_sample_data[:-1]):
            state_action_return.append((sample.state, sample.action, G))
            G = self.gamma * G + sample.reward

        state_action_return.reverse()

        # Update the Q-table
        # 更新Q表
        seen_state_action = set()
        for state, action, G in state_action_return:
            if (state, action) not in seen_state_action:
                self.visit[state][action] += 1

                # calculate incremental mean
                # 计算递增均值
                self.Q[state, action] = self.Q[state, action] + (G - self.Q[state, action]) / self.visit[state, action]
                seen_state_action.add((state, action))

        # Update policy
        # 更新策略
        for state in range(self.state_size):
            best_action = np.argmax(self.Q[state])
            self.policy[state] = best_action

        return

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        # To save the model, it can consist of multiple files,
        # and it is important to ensure that each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        model_file_path = f"{path}/model.ckpt-{str(id)}.npy"
        np.save(model_file_path, self.policy)
        self.logger.info(f"save model {model_file_path} successfully")

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        # When loading the model, you can load multiple files,
        # and it is important to ensure that each filename matches the one used during the save_model process.
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        model_file_path = f"{path}/model.ckpt-{str(id)}.npy"
        try:
            self.policy = np.load(model_file_path)
            self.logger.info(f"load model {model_file_path} successfully")
        except FileNotFoundError:
            self.logger.info(f"File {model_file_path} not found")
            exit(1)
