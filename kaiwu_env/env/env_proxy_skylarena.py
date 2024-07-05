
from kaiwu_env.env.base_env import BaseEnv
from kaiwu_env.zlib.zhelper import WORKER_FAULT_RESULT
from kaiwu_env.env.protocol import Parse_AIServerRequest, Parse_AIServerResponse
from kaiwu_env.utils.connect import ProxyHttpController
from kaiwu_env.zlib.rrrproxy import ReliaRRProxy
from kaiwu_env.zlib.rrrworker import worker_rrrproxy
from kaiwu_env.conf import yaml_rrrsvr
from random import randint
from time import sleep
from kaiwu_env.env.protocol import Parse_StepFrameReq
from kaiwu_env.env.protocol import Parse_AIServerRequest, Parse_AIServerResponse
from kaiwu_env.env.protocol import SkylarenaDataHandler


class EnvProxySkylarena(BaseEnv):
    def __init__(self, game_name, scene_name='default_scene', logger=None, monitor=None) -> None:
        self.game_name = game_name
        self.scene_name = scene_name
        self.http_tk = ProxyHttpController()
        proxy = ReliaRRProxy(
            yaml_rrrsvr.skylarena.server.url_ftend, 
            yaml_rrrsvr.skylarena.server.url_bkend, 
            flag_monitor=True if yaml_rrrsvr.skylarena.server.flag_monitor == 'true' else False,)
        proxy.run()
        worker_id = randint(0, 0x10000)
        self.worker = worker_rrrproxy(worker_id, yaml_rrrsvr.skylarena.worker.url_bkend, False, None, worker_id, )
        self.worker.send(None)
        self.data_handler = SkylarenaDataHandler(self.game_name, logger, monitor)
        
    def __recv_stepframereq(self):
        while True:
            byte_stepframe_req = self.worker.send(None) 
            if byte_stepframe_req == WORKER_FAULT_RESULT:
                continue
            return byte_stepframe_req
    
    def __send_stepframersp(self, byte_stepfram_rsp):
        self.worker.send(byte_stepfram_rsp)
    
    def reset(self, usr_conf={}):
        while not self.http_tk.start_game(usr_conf):
            sleep(3)
        # 从entity收到StepFrameReq请求
        byte_stepframe_req = self.__recv_stepframereq()

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

        # 将StepFrameRsp请求返回给entity
        self.__send_stepframersp(byte_stepframe_rsp)
        
        # 从entity收到StepFrameReq请求
        byte_stepframe_req = self.__recv_stepframereq()

        # 将StepFrameReq请求反序列化得到结构化数据, 逻辑由self.game_name决定（场景接入方实现)
        # game_id, frame_no, frame_state, terminated, truncated, game_info
        pb_stepframe_req = Parse_StepFrameReq.decode(self.game_name, byte_stepframe_req)
        
        # 将StepFrameReq的结构化数据转换成AIServerRequest并序列化，转换逻辑由self.game_name决定（场景接入方实现)
        byte_aisvr_req = self.data_handler.StepFrameReq2AISvrReq(pb_stepframe_req)

        # 将AIServerRequest反序列化后, 转换成用户调用env.reset或env.step期望获得的结构化数据
        # 转换逻辑由self.game_name决定（场景接入方实现）, 该函数逻辑与env.step返回的数据强相关, 需要接入方仔细实现
        self.game_id, self.frame_no, self.obs, self.score_info, self.terminated, self.truncated, self.env_info = Parse_AIServerRequest.decode(self.game_name, byte_aisvr_req)

        # 最后一帧需要单独处理
        # 游戏是否结束由game状态terminated,truncated决定
        if self.terminated or self.truncated or stop_game:
            # 用户env.step传入int或float类型的动作，转换成AIServerResponse并序列化, 
            # 转换逻辑由self.game_name决定（场景接入方实现）, 该函数逻辑与env.step返回的参数强相关, 需要接入方仔细实现
            byte_aisvr_rsp = Parse_AIServerResponse.encode(self.game_name, self.game_id, self.frame_no, act, self.terminated or self.truncated or stop_game)
            # 将AIServerResponse的结构化数据转换成StepFrameRsp并序列化，转换逻辑由self.game_name决定（场景接入方实现）
            byte_stepframe_rsp = self.data_handler.AISvrRsp2StepFrameRsp(pb_aisvr_rsp)
            # 将StepFrameRsp请求返回给entity
            self.__send_stepframersp(byte_stepframe_rsp)

        self.data_handler.step(pb_stepframe_req, pb_aisvr_rsp)

        return self.frame_no, self.obs, self.score_info, self.terminated, self.truncated, self.env_info

