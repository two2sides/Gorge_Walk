
from kaiwu_agent.utils.common_func import create_cls, attached


SampleData = create_cls("SampleData", state=None, action=None, reward=None)


ObsData = create_cls("ObsData", feature=None)


ActData = create_cls("ActData", act=None)


@attached
def observation_process(raw_obs):
    return ObsData(feature=int(raw_obs[0]))


@attached
def action_process(act_data):
    return act_data.act


@attached
def sample_process(list_game_data):
    return [SampleData(**i.__dict__) for i in list_game_data]


def reward_shaping(frame_no, score, terminated, truncated, obs):
    reward = 0

    return reward
