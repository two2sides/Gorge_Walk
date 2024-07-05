
from kaiwu_env.env.protocol import Parse_StepFrameRsp, Parse_StepFrameReq
from kaiwu_env.conf import yaml_rrrsvr
from kaiwu_env.zlib.p2pihc import P2PWorker
from kaiwu_env.utils.common_func import get_uuid
from random import randint
from pickle import dumps, loads
import logging
from kaiwu_env.utils.strategy import strategy_selector
from kaiwu_env.conf import yaml_arena

class EnvEntity:
    def __init__(self, env, game_name, scene_name='default_scene', logger=None, monitor=None) -> None:
        self.game_name = game_name
        self.env = strategy_selector("scene_wrapper", yaml_arena.gamecore_type, env)
        worker_id = randint(0, 0x10000)
        self.worker = P2PWorker(worker_id, yaml_rrrsvr.skylarena.server.url_ftend, logger)
        self.logger = logger if logger else logging.getLogger(__class__.__name__)
        self.monitor = monitor
        
    def __recv_uconf(self):
        while True:
            b_usr_conf = self.worker.recv_start_sess()
            if b_usr_conf == False:
                continue
            break
        return b_usr_conf
    
    def __send_stepframereq(self, byte_stepframe_req):
        return self.worker.send(byte_stepframe_req)

    def __recv_stepframersp(self):
        return self.worker.recv()

    def reset(self):
        # 返回false说明收到了SESS_STOP
        usr_conf = self.__recv_uconf()
        usr_conf = loads(usr_conf)

        game_id = get_uuid()
        game_id = yaml_arena.eval_game_id if yaml_arena.train_or_eval == "eval" else game_id

        # env.reset拿到游戏的信息
        game_id, frame_no, _frame_state, terminated, truncated, game_info = self.env.reset(game_id, usr_conf)

        # 将game.reset或game.step返回的每个游戏自定义的结构化数据转换成StepFrameReq并序列化
        # 转换逻辑由self.game_name决定（场景接入方实现）, 该函数逻辑与game.step返回的数据强相关, 需要接入方仔细实现
        byte_stepframe_req = Parse_StepFrameReq.encode(self.game_name, **{
            'game_id': game_id, 
            'frame_no': frame_no, 
            'frame_state': _frame_state, 
            'terminated': terminated,
            'truncated': truncated,
            'game_info': game_info})
        
        return self.__send_stepframereq(byte_stepframe_req)
    
    def step(self):

        # 收到StepFrameRsp
        byte_stepframe_rsp = self.__recv_stepframersp()
        if byte_stepframe_rsp == False:
            return False
        
        # 处理一下worker收到stop的情况
        if self.worker.does_recv_stop():
            self.worker.send(byte_stepframe_rsp)
            return False

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

        return self.__send_stepframereq(byte_stepframe_req)
        
    def run_once(self):
        flag = self.reset()
        if flag == False:
            return
        self.logger.debug('EnvEntity reset one game success')
        while True:
            flag = self.step()
            if flag == False:
                self.logger.info('EnvEntity run one episode done')
                # episode done 发送监控数据
                # if self.monitor:
                #     self.monitor.put_data({'alive': 'yes'})
                return
