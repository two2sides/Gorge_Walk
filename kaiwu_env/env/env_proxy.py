
from kaiwu_env.env.base_env import BaseEnv
from kaiwu_env.zlib.zhelper import WORKER_FAULT_RESULT
from kaiwu_env.env.protocol import Parse_AIServerRequest, Parse_AIServerResponse
from kaiwu_env.utils.connect import ProxyHttpController
from kaiwu_env.zlib.rrrproxy import ReliaRRProxy
from kaiwu_env.zlib.rrrworker import worker_rrrproxy
from kaiwu_env.conf import yaml_rrrsvr
from random import randint
from time import sleep


class EnvProxy(BaseEnv):
    def __init__(self, game_name, scene_name='default_scene', logger=None) -> None:
        self.game_name = game_name
        self.scene_name = scene_name
        self.http_tk = ProxyHttpController()
        proxy = ReliaRRProxy(
            yaml_rrrsvr.proxy.server.url_ftend, 
            yaml_rrrsvr.proxy.server.url_bkend, 
            flag_monitor=True if yaml_rrrsvr.proxy.server.flag_monitor == 'true' else False,)
        proxy.run()
        worker_id = randint(0, 0x10000)
        self.worker = worker_rrrproxy(worker_id, yaml_rrrsvr.proxy.worker.url_bkend, False, None, worker_id, )
        self.worker.send(None)
        
    def __recv_aisvrreq(self):
        while True:
            req = self.worker.send(None) 
            if req == WORKER_FAULT_RESULT:
                continue
            return req
    
    def __send_aisvrrsp(self, byte_aisvr_rsp):
        self.worker.send(byte_aisvr_rsp)
    
    def reset(self, usr_conf={}):
        while not self.http_tk.start_game(usr_conf):
            sleep(3)
        byte_aisvr_req = self.__recv_aisvrreq()
        # 将AIServerRequest反序列化后, 转换成用户调用env.reset或env.step期望获得的结构化数据
        # 转换逻辑由self.game_name决定（场景接入方实现）, 该函数逻辑与env.step返回的数据强相关, 需要接入方仔细实现
        self.game_id, self.frame_no, self.obs, self.score_info, self.terminated, self.truncated, self.env_info = Parse_AIServerRequest.decode(self.game_name, byte_aisvr_req)
        return self.obs
    
    def step(self, act, stop_game=False):
        # 用户env.step传入int或float类型的动作，转换成AIServerResponse并序列化, 
        # 转换逻辑由self.game_name决定（场景接入方实现）, 该函数逻辑与env.step返回的参数强相关, 需要接入方仔细实现
        byte_aisvr_rsp = Parse_AIServerResponse.encode(self.game_name, self.game_id, self.frame_no, act, self.terminated or self.truncated or stop_game)

        # 将AIServerResponse序列化后的数据发送给skylarena
        self.__send_aisvrrsp(byte_aisvr_rsp)

        # 从skylarena收到AIServerRequest请求
        byte_aisvr_req = self.__recv_aisvrreq()

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

