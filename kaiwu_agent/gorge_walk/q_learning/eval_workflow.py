from kaiwu_agent.utils.common_func import attached
from kaiwu_agent.agent.protocol.protocol import observation_process, action_process
import random, os
from kaiwu_agent.conf import yaml_gorge_walk_game as game_conf

def workflow(envs, agents, logger=None, monitor=None, treasure_cnt=5, treasure_random=False):    
    env, agent = envs[0], agents[0]

    # treasure_cnt = treasure_cnt if not os.environ.get('treasure_cnt') else int(os.environ.get('treasure_cnt'))
    # treasure_random = treasure_random if not os.environ.get("treasure_random") else bool(os.environ.get("treasure_random"))
    treasure_cnt = game_conf.treasure_num
    treasure_random = game_conf.treasure_random

    list_treasure_id = []
    
    if treasure_random:
        # 生成一个从0到9的列表
        numbers = list(range(10))
        # 从列表中随机取出3个元素
        if treasure_cnt:
            list_treasure_id = sorted(random.sample(numbers, treasure_cnt))
    
    else:
        if treasure_cnt:
            list_treasure_id = list(range(treasure_cnt))
    
    # 评估
    logger.info("Start Evaluation ...")

    EPISODE_CNT = 1
    total_score, win_cnt, treasure_cnt = 0, 0, 0
    for episode in range(EPISODE_CNT):
        # 固定游戏启动配置
        usr_conf = {"diy": {
            'start': [29, 9],
            #'end': [19, 14],
            'end': [11, 55],
            'treasure_id': list_treasure_id,
            #'treasure_id': -1,
            #'treasure_num': 5
            }
        }
        
        # 重置游戏, 并获取初始状态
        obs = env.reset(usr_conf=usr_conf)
        
        # 特征处理
        obs_data = observation_process(obs)
        
        # 游戏循环
        done = False
        while not done:
            # Agent 进行推理, 获取下一帧的预测动作
            act_data = agent.exploit(list_obs_data=[obs_data])[0]
            
            # ActData 解包成动作
            act = action_process(act_data)
            
            # 与环境交互, 执行动作, 获取下一步的状态
            frame_no, _obs, score, terminated, truncated, game_info = env.step(act)
            
            # 特征处理
            obs_data = observation_process(_obs)
            
            # 判断游戏结束, 并更新胜利次数
            done = terminated or truncated
            if terminated:
                win_cnt += 1
                
            # 更新总奖励和状态
            total_score += score
        
        # 更新宝箱收集数量
        treasure_cnt += game_info.treasure_count
        
        # 打印评估进度
        if episode % 10 == 0 and episode > 0:
            logger.info(f"Episode: {episode + 1} ...")
            
    # 打印评估结果
    logger.info(f"Average Total Score: {total_score / EPISODE_CNT}")
    logger.info(f"Average Treasure Collected: {treasure_cnt / EPISODE_CNT}")
    logger.info(f"Success Rate : {win_cnt / EPISODE_CNT}")