

from kaiwu_env.env.base_env import BaseEnv
from kaiwu_env.zlib.zhelper import WORKER_FAULT_RESULT
from kaiwu_env.env.protocol import Parse_AIServerResponse
from kaiwu_env.env.protocol import Parse_StepFrameReq
from kaiwu_env.env.protocol import SkylarenaDataHandler
from kaiwu_env.zlib.p2pihc import P2PWorker
from kaiwu_env.zlib.p2pihc import P2PClient
from kaiwu_env.conf import yaml_rrrsvr
from random import randint
import logging
from pickle import loads


class EnvSkylarena(BaseEnv):
    def __init__(self, game_name, scene_name='default_scene', logger=None, monitor=None) -> None:
        self.game_name = game_name
        self.scene_name = scene_name
        self.data_handler = SkylarenaDataHandler(self.game_name, logger, monitor)
        self.episode_counter = 0
        worker_id = randint(0, 0x10000)
        self.worker = P2PWorker(worker_id, yaml_rrrsvr.skylarena.server.url_ftend, logger)
        self.client = P2PClient(1, yaml_rrrsvr.proxy.client.url_ftend, logger)
        self.logger = logger if logger else logging.getLogger(__class__.__name__)
        self.usr_conf = None
    
    def __recv_req_send_uconf(self):
        self.episode_counter += 1
        while True:
            req = self.worker.recv_start_sess()
            if req == False:
                continue
            break
        while True:
            b_usr_conf = self.client.start_sess(req)
            if b_usr_conf == False:
                continue
            break
        self.usr_conf = loads(b_usr_conf)
        self.worker.send(b_usr_conf)
        
    def __recv_stepframereq(self):
        return self.worker.recv()
    
    def __send_aisvrreq_recv_aisvrrsp(self, byte_aisvr_req):
        does_worker_recv_stop = self.worker.does_recv_stop()
        if does_worker_recv_stop == False:
            int_seq_no, cmd, byte_aisvrrsp = self.client.update_sess(byte_aisvr_req)
        else:
            byte_aisvrrsp = self.client.stop_sess(byte_aisvr_req)
        return byte_aisvrrsp 

    def __send_stepframersp(self, byte_stepfram_rsp):
        return self.worker.send(byte_stepfram_rsp)
        
    def reset(self):
        self.__recv_req_send_uconf()
        self.data_handler.reset(self.usr_conf)
    
    def step(self):
        # 从entity收到StepFrameReq请求
        byte_stepframe_req = self.__recv_stepframereq()
        if byte_stepframe_req == False:
            return True

        # 将StepFrameReq请求反序列化得到结构化数据, 逻辑由self.game_name决定（场景接入方实现)
        # game_id, frame_no, frame_state, terminated, truncated, game_info
        pb_stepframe_req = Parse_StepFrameReq.decode(self.game_name, byte_stepframe_req)

        # 将StepFrameReq的结构化数据转换成AIServerRequest并序列化，转换逻辑由self.game_name决定（场景接入方实现)
        byte_aisvr_req = self.data_handler.StepFrameReq2AISvrReq(pb_stepframe_req)

        # 发送AIServerRequest并收到AIServerResponse
        byte_aisvr_rsp = self.__send_aisvrreq_recv_aisvrrsp(byte_aisvr_req)
        if byte_aisvr_rsp == False:
            return True

        # 将AIServerResponse反序列化得到结构化数据, 逻辑由self.game_name决定（场景接入方实现)
        # 返回的pb可以解析成: game_id, frame_no, action, stop_game  
        pb_aisvr_rsp = Parse_AIServerResponse.decode(self.game_name, byte_aisvr_rsp)

        # 将AIServerResponse的结构化数据转换成StepFrameRsp并序列化，转换逻辑由self.game_name决定（场景接入方实现）
        byte_stepframe_rsp = self.data_handler.AISvrRsp2StepFrameRsp(pb_aisvr_rsp)

        # 集合一次通信所有的信息, 由data_handler.step进行处理
        self.data_handler.step(pb_stepframe_req, pb_aisvr_rsp)

        # 将StepFrameRsp请求返回给entity, 处理一下worker收到stop的情况
        flag = self.__send_stepframersp(byte_stepframe_rsp)
        if flag == False:
            return True

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
            

