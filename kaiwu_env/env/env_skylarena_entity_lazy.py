

from kaiwu_env.env.base_env import BaseEnv
from kaiwu_env.zlib.zhelper import WORKER_FAULT_RESULT
from kaiwu_env.env.protocol import Parse_AIServerResponse
from kaiwu_env.env.protocol import Parse_StepFrameReq
from kaiwu_env.env.protocol import SkylarenaDataHandler
from kaiwu_env.zlib.p2pihc import P2PWorker
from kaiwu_env.zlib.p2pihc import P2PClient
from kaiwu_env.conf import yaml_rrrsvr
from kaiwu_env.utils.common_func import get_uuid
from random import randint
from pickle import dumps, loads
import logging
from kaiwu_env.env.protocol import Parse_StepFrameRsp
from kaiwu_env.utils.strategy import strategy_selector
from kaiwu_env.conf import yaml_arena


class EnvSkylarenaEntity(BaseEnv):
    def __init__(self, env, game_name, scene_name='default_scene', logger=None, monitor=None) -> None:
        self.game_name = game_name
        self.scene_name = scene_name
        self.env = strategy_selector("scene_wrapper", yaml_arena.gamecore_type, env)
        self.data_handler = SkylarenaDataHandler(self.game_name, logger, monitor)
        worker_id = randint(0, 0x10000)
        self.worker = P2PWorker(worker_id, yaml_rrrsvr.proxy.server.url_ftend, logger)
        self.logger = logger if logger else logging.getLogger(__class__.__name__)
        self.monitor = monitor
    
    def __recv_uconf_get_stepframereq(self):
        while True:
            b_usr_conf = self.worker.recv_start_sess()
            if b_usr_conf == False:
                continue
            break

        usr_conf = loads(b_usr_conf)
        game_id = get_uuid()
        game_id = yaml_arena.eval_game_id if yaml_arena.train_or_eval == "eval" else game_id

        # env.reset拿到游戏的信息
        game_id, frame_no, _frame_state, terminated, truncated, game_info = self.env.reset(game_id, usr_conf)

        # 返回false说明收到了SESS_STOP
        self.data_handler.reset(usr_conf)

        # 将game.reset或game.step返回的每个游戏自定义的结构化数据转换成StepFrameReq并序列化
        # 转换逻辑由self.game_name决定（场景接入方实现）, 该函数逻辑与game.step返回的数据强相关, 需要接入方仔细实现
        byte_stepframe_req = Parse_StepFrameReq.encode(self.game_name, **{
            'game_id': game_id, 
            'frame_no': frame_no, 
            'frame_state': _frame_state, 
            'terminated': terminated,
            'truncated': truncated,
            'game_info': game_info})
        return byte_stepframe_req
            
    def __send_aisvrreq(self, byte_aisvr_req):
        return self.worker.send(byte_aisvr_req)

    def __recv_aisvrrsp(self):
        return self.worker.recv()


    def __send__stepframersp_recv_stepframereq(self, byte_stepframe_rsp):
        int_seq_no, cmd, byte_stepframe_req = self.client.update_sess(byte_stepframe_rsp)
        return byte_stepframe_req 

        
    def reset(self):

        # 不会发生收不到stepframereq的情况, 该模式包含一个env
        byte_stepframe_req = self.__recv_uconf_get_stepframereq()
        # 将StepFrameReq请求反序列化得到结构化数据, 逻辑由self.game_name决定（场景接入方实现)
        # game_id, frame_no, frame_state, terminated, truncated, game_info
        pb_stepframe_req = Parse_StepFrameReq.decode(self.game_name, byte_stepframe_req)

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
            self.worker.send(byte_aisvr_rsp)
            return False

        # 将AIServerResponse反序列化得到结构化数据, 逻辑由self.game_name决定（场景接入方实现)
        # 返回的pb可以解析成: game_id, frame_no, action, stop_game  
        pb_aisvr_rsp = Parse_AIServerResponse.decode(self.game_name, byte_aisvr_rsp)

        # 将AIServerResponse的结构化数据转换成StepFrameRsp并序列化，转换逻辑由self.game_name决定（场景接入方实现）
        byte_stepframe_rsp = self.data_handler.AISvrRsp2StepFrameRsp(pb_aisvr_rsp)

        # 将StepFrameRsp反序列化得到pb数据，将pb数据转换成能被game.step接受的输入(每个游戏自定义的结构化数据)
        # 转换逻辑由self.game_name决定（场景接入方实现), 该函数逻辑与game.step的参数强相关, 需要接入方仔细实现
        game_id, frame_no, command, stop_game = Parse_StepFrameRsp.decode(self.game_name, byte_stepframe_rsp)

        # 调用真实的游戏step, 推动一帧
        game_id, frame_no, _frame_state, terminated, truncated, game_info = self.env.step(game_id, frame_no, command, stop_game)

        # 将game.reset或game.step返回的每个游戏自定义的结构化数据转换成StepFrameReq并序列化
        # 转换逻辑由self.game_name决定（场景接入方实现）, 该函数逻辑与game.step返回的数据强相关, 需要接入方仔细实现
        byte_stepframe_req = Parse_StepFrameReq.encode(self.game_name, **{
            'game_id': game_id, 
            'frame_no': frame_no, 
            'frame_state': _frame_state, 
            'terminated': terminated,
            'truncated': truncated,
            'game_info': game_info})

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
                if self.monitor:
                    self.monitor.put_data({'alive': 'yes'})
                break
            

