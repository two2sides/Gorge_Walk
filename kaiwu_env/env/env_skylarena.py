

from kaiwu_env.env.base_env import BaseEnv
from kaiwu_env.zlib.zhelper import WORKER_FAULT_RESULT
from kaiwu_env.env.protocol import Parse_AIServerResponse
from kaiwu_env.env.protocol import Parse_StepFrameReq
from kaiwu_env.env.protocol import SkylarenaDataHandler
from kaiwu_env.utils.connect import SkylarenaHttpController
from kaiwu_env.zlib.rrrproxy import ReliaRRProxy
from kaiwu_env.zlib.rrrworker import worker_rrrproxy
from kaiwu_env.zlib.rrrclient import client_for_arena
from kaiwu_env.conf import yaml_rrrsvr
from random import randint


class EnvSkylarena(BaseEnv):
    def __init__(self, game_name, scene_name='default_scene', logger=None, monitor=None) -> None:
        self.game_name = game_name
        self.scene_name = scene_name
        self.http_tk = SkylarenaHttpController()
        self.data_handler = SkylarenaDataHandler(self.game_name, logger, monitor)
        proxy = ReliaRRProxy(
            yaml_rrrsvr.skylarena.server.url_ftend, 
            yaml_rrrsvr.skylarena.server.url_bkend, 
            flag_monitor=True if yaml_rrrsvr.skylarena.server.flag_monitor == 'true' else False,)
        proxy.run()
        worker_id = randint(0, 0x10000)
        self.worker = worker_rrrproxy(worker_id, yaml_rrrsvr.skylarena.worker.url_bkend, False, None, worker_id, )
        self.worker.send(None)

        self.sender = client_for_arena(1, yaml_rrrsvr.proxy.client.url_ftend)
        self.sender.send(None)
        
    def __recv_stepframereq(self):
        while True:
            byte_stepframe_req = self.worker.send(None) 
            if byte_stepframe_req == WORKER_FAULT_RESULT:
                continue
            return byte_stepframe_req
    
    def __send_aisvrreq_recv_aisvrrsp(self, byte_aisvr_req):
        return self.sender.send(byte_aisvr_req)

    def __send_stepframersp(self, byte_stepfram_rsp):
        self.worker.send(byte_stepfram_rsp)
        
    def reset(self):
        self.http_tk.run_once()
        self.data_handler.reset()
    
    def step(self):
        # 从entity收到StepFrameReq请求
        byte_stepframe_req = self.__recv_stepframereq()

        # 将StepFrameReq请求反序列化得到结构化数据, 逻辑由self.game_name决定（场景接入方实现)
        # game_id, frame_no, frame_state, terminated, truncated, game_info
        pb_stepframe_req = Parse_StepFrameReq.decode(self.game_name, byte_stepframe_req)

        # 将StepFrameReq的结构化数据转换成AIServerRequest并序列化，转换逻辑由self.game_name决定（场景接入方实现)
        byte_aisvr_req = self.data_handler.StepFrameReq2AISvrReq(pb_stepframe_req)

        # 发送AIServerRequest并收到AIServerResponse
        byte_aisvr_rsp = self.__send_aisvrreq_recv_aisvrrsp(byte_aisvr_req)

        # 将AIServerResponse反序列化得到结构化数据, 逻辑由self.game_name决定（场景接入方实现)
        # 返回的pb可以解析成: game_id, frame_no, action, stop_game  
        pb_aisvr_rsp = Parse_AIServerResponse.decode(self.game_name, byte_aisvr_rsp)

        # 将AIServerResponse的结构化数据转换成StepFrameRsp并序列化，转换逻辑由self.game_name决定（场景接入方实现）
        byte_stepframe_rsp = self.data_handler.AISvrRsp2StepFrameRsp(pb_aisvr_rsp)

        # 将StepFrameRsp请求返回给entity
        self.__send_stepframersp(byte_stepframe_rsp)

        # 集合一次通信所有的信息, 由data_handler.step进行处理
        self.data_handler.step(pb_stepframe_req, pb_aisvr_rsp)

        # 返回游戏结束条件, game状态terminated,truncated和aisvr状态game_over
        return pb_stepframe_req.terminated or pb_stepframe_req.truncated or pb_aisvr_rsp.stop_game
    
    def run_once(self):
        self.reset()
        while True:
            # 游戏是否结束由game状态terminated,truncated和aisvr状态game_over决定
            done = self.step()
            if done:
                # 如果data_handler要在游戏结束时做特殊处理, 需要在finish中实现
                self.data_handler.finish()
                break
            

