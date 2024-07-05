
from kaiwu_env.env.base_env import BaseEnv
from kaiwu_env.zlib.zhelper import WORKER_FAULT_RESULT, RESET_RETRIES
from kaiwu_env.env.protocol import Parse_AIServerRequest, Parse_AIServerResponse
from kaiwu_env.zlib.p2pihc import P2PClient
from kaiwu_env.conf import yaml_rrrsvr
from random import randint
from time import sleep
from pickle import dumps, loads
import logging
from kaiwu_env.conf import yaml_arena


class EnvProxy(BaseEnv):
    def __init__(self, game_name, scene_name='default_scene', logger=None, monitor=None) -> None:
        self.game_name = game_name
        self.scene_name = scene_name
        client_id = randint(0, 0x10000)
        self.client = P2PClient(client_id, yaml_arena.skylarena_url, logger)
        self.does_game_start = False
        self.logger = logger if logger else logging.getLogger(__class__.__name__)

    def __send_uconf_recv_aisvrreq(self, b_usr_conf):
        # 会一直循环，若超过次数返回False, 需要容灾
        for _ in range(RESET_RETRIES):
            byte_aisvr_req = self.client.start_sess(b_usr_conf)
            if byte_aisvr_req == False:
                continue
            return byte_aisvr_req
        return False
    
    def __send_aisvrrsp_recv_aisvrreq(self, byte_aisvr_rsp):
        int_seq_no, cmd, byte_aisvr_req = self.client.update_sess(byte_aisvr_rsp)
        return byte_aisvr_req


    def reset(self, usr_conf={}):
        b_usr_conf = dumps(usr_conf)
        # 保证已经开始的游戏在start前必须有一个stop, 否则进入容灾
        if self.does_game_start:
            self.client.stop_sess()
            self.does_game_start = False
        # 1. 如果发生收不到byte_aisvr_req的情况, 会一直循环直到收到, 不需要容灾
        # 2. 如果发生收不到byte_aisvr_req的情况, 会一直循环，若超过次数返回False, 需要容灾, 采用
        byte_aisvr_req = self.__send_uconf_recv_aisvrreq(b_usr_conf)
        if byte_aisvr_req == False:
            return False
        # 保证已经开始的游戏被标记为True
        self.does_game_start = True

        # 将AIServerRequest反序列化后, 转换成用户调用env.reset或env.step期望获得的结构化数据
        # 转换逻辑由self.game_name决定（场景接入方实现）, 该函数逻辑与env.step返回的数据强相关, 需要接入方仔细实现
        self.game_id, self.frame_no, self.obs, self.score_info, self.terminated, self.truncated, self.env_info = Parse_AIServerRequest.decode(self.game_name, byte_aisvr_req)
        self.logger.debug('EnvProxy reset one game success')
        return self.obs
    
    def step(self, act, stop_game=False):
        # 用户env.step传入int或float类型的动作，转换成AIServerResponse并序列化, 
        # 转换逻辑由self.game_name决定（场景接入方实现）, 该函数逻辑与env.step返回的参数强相关, 需要接入方仔细实现
        byte_aisvr_rsp = Parse_AIServerResponse.encode(self.game_name, self.game_id, self.frame_no, act, self.terminated or self.truncated or stop_game)

        # 将AIServerResponse序列化后的数据发送给skylarena
        # 从skylarena收到AIServerRequest请求
        byte_aisvr_req = self.__send_aisvrrsp_recv_aisvrreq(byte_aisvr_rsp)

        # 如果发生收不到byte_aisvr_req的情况, 告警并且告知用户truncated==True, 需要重新reset
        if byte_aisvr_req == False:
            # self.frame_no, self.obs, self.score_info, self.terminated, self.truncated, self.env_info
            return None, None, None, None, True, None
            
        # 将AIServerRequest反序列化后, 转换成用户调用env.reset或env.step期望获得的结构化数据
        # 转换逻辑由self.game_name决定（场景接入方实现）, 该函数逻辑与env.step返回的数据强相关, 需要接入方仔细实现
        self.game_id, self.frame_no, self.obs, self.score_info, self.terminated, self.truncated, self.env_info = Parse_AIServerRequest.decode(self.game_name, byte_aisvr_req)

        return self.frame_no, self.obs, self.score_info, self.terminated, self.truncated, self.env_info

