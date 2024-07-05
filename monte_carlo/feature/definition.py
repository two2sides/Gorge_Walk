#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project :gorge_walk
@File    :definition.py
@Author  :kaiwu
@Date    :2022/12/15 22:50

"""


from kaiwu_agent.utils.common_func import create_cls, attached


SampleData = create_cls("SampleData", state=None, action=None, reward=None)


ObsData = create_cls("ObsData", feature=None)


ActData = create_cls("ActData", act=None)


@attached
def observation_process(raw_obs):
    # 默认仅使用位置信息作为特征, 如进行额外特征处理, 则需要对算法的Policy结构, predict, exploit, learn进行相应的改动
    # pos = int(raw_obs[0])
    # treasure_status = [int(item) for item in raw_obs[-10:]]
    # state = 1024 * pos + sum([treasure_status[i] * (2**i) for i in range(10)])
    # return ObsData(feature=int(state))

    return ObsData(feature=int(raw_obs[0]))


@attached
def action_process(act_data):
    return act_data.act


@attached
def sample_process(list_game_data):
    return [SampleData(**i.__dict__) for i in list_game_data]


def reward_shaping(frame_no, score, terminated, truncated, obs):
    reward = 0

    # Using the environment's score as the reward
    # 奖励1. 使用环境的得分作为奖励
    reward += score

    # Penalty for the number of steps
    # 奖励2. 步数惩罚
    if not terminated:
        reward += -1

    # The reward for being close to the finish line
    # 奖励3. 靠近终点的奖励:

    # The reward for being close to the treasure chest (considering only the nearest one)
    # 奖励4. 靠近宝箱的奖励(只考虑最近的那个宝箱)

    return reward
