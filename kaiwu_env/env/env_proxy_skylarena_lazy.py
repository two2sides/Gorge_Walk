
from kaiwu_env.env.base_env import BaseEnv
from kaiwu_env.zlib.zhelper import WORKER_FAULT_RESULT
from kaiwu_env.env.protocol import Parse_AIServerRequest, Parse_AIServerResponse
from kaiwu_env.zlib.p2pihc import P2PClient
from kaiwu_env.env.protocol import Parse_AIServerResponse
from kaiwu_env.env.protocol import Parse_StepFrameReq
from kaiwu_env.env.protocol import SkylarenaDataHandler
from kaiwu_env.conf import yaml_rrrsvr
from random import randint
from time import sleep
from pickle import dumps, loads
import logging


class EnvProxySkylarena(BaseEnv):
    def __init__(self, game_name, scene_name='default_scene', logger=None, monitor=None) -> None:
        self.game_name = game_name
        self.scene_name = scene_name
        self.data_handler = SkylarenaDataHandler(self.game_name, logger, monitor)
        client_id = randint(0, 0x10000)
        self.client = P2PClient(client_id, yaml_rrrsvr.skylarena.client.url_ftend, logger)
        self.does_game_start = False
        self.logger = logger if logger else logging.getLogger(__class__.__name__)

    def __send_uconf_recv_stepframereq(self, b_usr_conf):
        while True:
            byte_stepframe_req = self.client.start_sess(b_usr_conf)
            if byte_stepframe_req == False:
                continue
            return byte_stepframe_req
    
    def __send__stepframersp_recv_stepframereq(self, byte_stepframe_rsp):
        int_seq_no, cmd, byte_stepframe_req = self.client.update_sess(byte_stepframe_rsp)
        return byte_stepframe_req 


    def reset(self, usr_conf={}):
        b_usr_conf = dumps(usr_conf)
        # 保证已经开始的游戏在start前必须有一个stop, 否则进入容灾
        if self.does_game_start:
            self.client.stop_sess()
            self.does_game_start = False
        # 如果发生收不到stepframereq的情况, 会一直循环直到收到
        byte_stepframe_req = self.__send_uconf_recv_stepframereq(b_usr_conf)
        # 保证已经开始的游戏被标记为True
        self.does_game_start = True
        # 将StepFrameReq请求反序列化得到结构化数据, 逻辑由self.game_name决定（场景接入方实现)
        # game_id, frame_no, frame_state, terminated, truncated, game_info
        pb_stepframe_req = Parse_StepFrameReq.decode(self.game_name, byte_stepframe_req)

        # 将StepFrameReq的结构化数据转换成AIServerRequest并序列化，转换逻辑由self.game_name决定（场景接入方实现)
        byte_aisvr_req = self.data_handler.StepFrameReq2AISvrReq(pb_stepframe_req)

        # 将AIServerRequest反序列化后, 转换成用户调用env.reset或env.step期望获得的结构化数据
        # 转换逻辑由self.game_name决定（场景接入方实现）, 该函数逻辑与env.step返回的数据强相关, 需要接入方仔细实现
        self.game_id, self.frame_no, self.obs, self.score_info, self.terminated, self.truncated, self.env_info = Parse_AIServerRequest.decode(self.game_name, byte_aisvr_req)
        return self.obs
    
    def step(self, act, stop_game=False):
        # 用户env.step传入int或float类型的动作，转换成AIServerResponse并序列化, 
        # 转换逻辑由self.game_name决定（场景接入方实现）, 该函数逻辑与env.step返回的参数强相关, 需要接入方仔细实现
        byte_aisvr_rsp = Parse_AIServerResponse.encode(self.game_name, self.game_id, self.frame_no, act, self.terminated or self.truncated or stop_game)

        # 将AIServerResponse反序列化得到结构化数据, 逻辑由self.game_name决定（场景接入方实现)
        # 返回的pb可以解析成: game_id, frame_no, action, stop_game  
        pb_aisvr_rsp = Parse_AIServerResponse.decode(self.game_name, byte_aisvr_rsp)

        # 将AIServerResponse的结构化数据转换成StepFrameRsp并序列化，转换逻辑由self.game_name决定（场景接入方实现）
        byte_stepframe_rsp = self.data_handler.AISvrRsp2StepFrameRsp(pb_aisvr_rsp)

        # StepFrameRsp请求返回给entity, 从entity收到StepFrameReq请求
        byte_stepframe_req = self.__send__stepframersp_recv_stepframereq(byte_stepframe_rsp)
        if byte_stepframe_req == False:
            return False
        
        # 将StepFrameReq请求反序列化得到结构化数据, 逻辑由self.game_name决定（场景接入方实现)
        # game_id, frame_no, frame_state, terminated, truncated, game_info
        pb_stepframe_req = Parse_StepFrameReq.decode(self.game_name, byte_stepframe_req)

        # 将StepFrameReq的结构化数据转换成AIServerRequest并序列化，转换逻辑由self.game_name决定（场景接入方实现)
        byte_aisvr_req = self.data_handler.StepFrameReq2AISvrReq(pb_stepframe_req)

        # 如果发生收不到byte_aisvr_req的情况, 告警并且告知用户truncated==True, 需要重新reset
        if byte_aisvr_req == False:
            # self.frame_no, self.obs, self.score_info, self.terminated, self.truncated, self.env_info
            return None, None, None, None, True, None
            
        # 将AIServerRequest反序列化后, 转换成用户调用env.reset或env.step期望获得的结构化数据
        # 转换逻辑由self.game_name决定（场景接入方实现）, 该函数逻辑与env.step返回的数据强相关, 需要接入方仔细实现
        self.game_id, self.frame_no, self.obs, self.score_info, self.terminated, self.truncated, self.env_info = Parse_AIServerRequest.decode(self.game_name, byte_aisvr_req)

        self.data_handler.step(pb_stepframe_req, pb_aisvr_rsp)

        return self.frame_no, self.obs, self.score_info, self.terminated, self.truncated, self.env_info

