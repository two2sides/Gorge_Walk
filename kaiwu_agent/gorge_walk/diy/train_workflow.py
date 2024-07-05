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
    env, agent = envs[0], agents[0]
    # model saving
    # 保存模型
    agent.save_model()

    return