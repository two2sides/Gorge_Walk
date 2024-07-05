

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
        self.does_game_start = False
        worker_id = randint(0, 0x10000)
        self.worker = P2PWorker(worker_id, yaml_rrrsvr.proxy.server.url_ftend, logger)
        self.client = P2PClient(1, yaml_rrrsvr.skylarena.client.url_ftend, logger)
        self.logger = logger if logger else logging.getLogger(__class__.__name__)
        self.monitor = monitor
        self.usr_conf = None
        
    def __send_uconf_recv_stepframereq(self):
        while True:
            b_usr_conf = self.worker.recv_start_sess()
            if b_usr_conf == False:
                continue
            break
        while True:
            byte_stepframe_req = self.client.start_sess(b_usr_conf)
            if byte_stepframe_req == False:
                continue
            break
        self.usr_conf = loads(b_usr_conf)
        return byte_stepframe_req
            
    def __send_aisvrreq(self, byte_aisvr_req):
        return self.worker.send(byte_aisvr_req)

    def __recv_aisvrrsp(self):
        return self.worker.recv()


    def __send__stepframersp_recv_stepframereq(self, byte_stepframe_rsp):
        int_seq_no, cmd, byte_stepframe_req = self.client.update_sess(byte_stepframe_rsp)
        return byte_stepframe_req 

        
    def reset(self):
        # 保证已经开始的游戏在start前必须有一个stop, 否则进入容灾
        if self.does_game_start:
            self.client.stop_sess()
            self.does_game_start = False
        # 如果发生收不到stepframereq的情况, 会一直循环直到收到
        byte_stepframe_req = self.__send_uconf_recv_stepframereq()
        # 保证已经开始的游戏被标记为True
        self.does_game_start = True
        # 将StepFrameReq请求反序列化得到结构化数据, 逻辑由self.game_name决定（场景接入方实现)
        # game_id, frame_no, frame_state, terminated, truncated, game_info
        pb_stepframe_req = Parse_StepFrameReq.decode(self.game_name, byte_stepframe_req)

        # 返回false说明收到了SESS_STOP
        self.data_handler.reset(self.usr_conf)

        # 将StepFrameReq的结构化数据转换成AIServerRequest并序列化，转换逻辑由self.game_name决定（场景接入方实现)
        byte_aisvr_req = self.data_handler.StepFrameReq2AISvrReq(pb_stepframe_req)

        # 发送AIServerRequest完成一次交互流程
        return self.__send_aisvrreq(byte_aisvr_req)
    
    def step(self):
        # 返回false说明worker收到了SESS_STOP或者client收到了False
        # 收到AIServerResponse
        byte_aisvr_rsp = self.__recv_aisvrrsp()
        if byte_aisvr_rsp == False:
            return False
        
        # 处理一下worker收到stop的情况
        if self.worker.does_recv_stop():
            self.worker.send(b'STOP_SESS_RESPONSE')
            return False

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

        # 在step的最后处理handler.step
        self.data_handler.step(pb_stepframe_req, pb_aisvr_rsp)

        # 发送AIServerRequest
        return self.__send_aisvrreq(byte_aisvr_req)

    
    def run_once(self):
        flag = self.reset()
        if flag == False:
            return
        self.logger.debug('EnvSkylarena reset one game success')
        while True:
            # 游戏是否结束由game状态terminated,truncated和aisvr状态game_over决定
            flag = self.step()
            if flag == False:
                # 如果data_handler要在游戏结束时做特殊处理, 需要在finish中实现
                self.data_handler.finish()
                self.logger.info('EnvSkylarena run one episode done')
                # episode done 发送监控数据
                # if self.monitor:
                #     self.monitor.put_data({'alive': 'yes'})
                break
            

