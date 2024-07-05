
from kaiwu_env.env.base_env import BaseEnv
from kaiwu_env.zlib.zhelper import WORKER_FAULT_RESULT
from kaiwu_env.env.protocol import Parse_AIServerRequest, Parse_AIServerResponse
from kaiwu_env.zlib.p2pihc import P2PWorker
from kaiwu_env.conf import yaml_rrrsvr
from random import randint
from time import sleep
from pickle import dumps, loads
import logging


class EnvProxy(BaseEnv):
    def __init__(self, game_name, scene_name='default_scene', logger=None, monitor=None) -> None:
        self.game_name = game_name
        self.scene_name = scene_name
        worker_id = randint(0, 0x10000)
        self.worker = P2PWorker(worker_id, yaml_rrrsvr.proxy.server.url_ftend, logger)
        self.episode_counter = 0
        self.logger = logger if logger else logging.getLogger(__class__.__name__)
        
    def __recv_req_send_uconf(self, b_usr_conf):
        self.episode_counter += 1
        while True:
            req = self.worker.recv_start_sess()
            if req == False:
                continue
            break
        return self.worker.send(b_usr_conf)

    def __recv_aisvrreq(self):
        return self.worker.recv()
    
    def __send_aisvrrsp(self, byte_aisvr_rsp):
        return self.worker.send(byte_aisvr_rsp)
    
    def reset(self, usr_conf={}):
        b_usr_conf = dumps(usr_conf)
        self.__recv_req_send_uconf(b_usr_conf)

        byte_aisvr_req = self.__recv_aisvrreq()
        if byte_aisvr_req == False:
            return None

        # 将AIServerRequest反序列化后, 转换成用户调用env.reset或env.step期望获得的结构化数据
        # 转换逻辑由self.game_name决定（场景接入方实现）, 该函数逻辑与env.step返回的数据强相关, 需要接入方仔细实现
        self.game_id, self.frame_no, self.obs, self.score_info, self.terminated, self.truncated, self.env_info = Parse_AIServerRequest.decode(self.game_name, byte_aisvr_req)
        return self.obs
    
    def step(self, act, stop_game=False):
        # 用户env.step传入int或float类型的动作，转换成AIServerResponse并序列化, 
        # 转换逻辑由self.game_name决定（场景接入方实现）, 该函数逻辑与env.step返回的参数强相关, 需要接入方仔细实现
        byte_aisvr_rsp = Parse_AIServerResponse.encode(self.game_name, self.game_id, self.frame_no, act, self.terminated or self.truncated or stop_game)

        # 将AIServerResponse序列化后的数据发送给skylarena
        flag = self.__send_aisvrrsp(byte_aisvr_rsp)
        if flag == False:
            return None, None, None, None, True, None

        # 从skylarena收到AIServerRequest请求
        byte_aisvr_req = self.__recv_aisvrreq()
        if byte_aisvr_req == False:
            # self.frame_no, self.obs, self.score_info, self.terminated, self.truncated, self.env_info
            return None, None, None, None, True, None

        # 将AIServerRequest反序列化后, 转换成用户调用env.reset或env.step期望获得的结构化数据
        # 转换逻辑由self.game_name决定（场景接入方实现）, 该函数逻辑与env.step返回的数据强相关, 需要接入方仔细实现
        self.game_id, self.frame_no, self.obs, self.score_info, self.terminated, self.truncated, self.env_info = Parse_AIServerRequest.decode(self.game_name, byte_aisvr_req)

        # 最后一帧需要单独处理
        # 游戏是否结束由game状态terminated,truncated决定
        if self.terminated or self.truncated or stop_game:
            # 用户env.step传入int或float类型的动作，转换成AIServerResponse并序列化, 
            # 转换逻辑由self.game_name决定（场景接入方实现）, 该函数逻辑与env.step返回的参数强相关, 需要接入方仔细实现
            byte_aisvr_rsp = Parse_AIServerResponse.encode(self.game_name, self.game_id, self.frame_no, act, self.terminated or self.truncated or stop_game)
            self.__send_aisvrrsp(byte_aisvr_rsp)

        return self.frame_no, self.obs, self.score_info, self.terminated, self.truncated, self.env_info

