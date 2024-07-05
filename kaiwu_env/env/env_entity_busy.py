
from kaiwu_env.env.protocol import Parse_StepFrameRsp, Parse_StepFrameReq
from kaiwu_env.utils.connect import EntityHttpController
from kaiwu_env.conf import yaml_rrrsvr
from kaiwu_env.zlib.p2pihc import P2PClient
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
        client_id = randint(0, 0x10000)
        self.client = P2PClient(client_id, yaml_rrrsvr.skylarena.client.url_ftend, logger)
        self.episode_counter = 0
        self.logger = logger if logger else logging.getLogger(__class__.__name__)
    
    def __send_req_recv_usrconf(self):
        self.episode_counter += 1
        while True:
            b_usr_conf = self.client.start_sess(b'please pass usr conf')
            if b_usr_conf == False:
                continue
            return b_usr_conf
        
    def __send_stepframereq_recv_stepframersp(self, byte_stepframe_req):
        int_seq_no, cmd, byte_aisvr_req = self.client.update_sess(byte_stepframe_req)
        return byte_aisvr_req

    def reset(self):
        # 发送请求，收到usrconf
        b_usr_conf = self.__send_req_recv_usrconf()
        usr_conf = loads(b_usr_conf)

        game_id = get_uuid()
        game_id = yaml_arena.eval_game_id if yaml_arena.train_or_eval == "eval" else game_id

        # env.reset拿到游戏的信息
        game_id, frame_no, _frame_state, terminated, truncated, game_info = self.env.reset(game_id, usr_conf)
        return game_id, frame_no, _frame_state, terminated, truncated, game_info
    
    def step(self, game_id, frame_no, _frame_state, terminated, truncated, game_info):
        # 将game.reset或game.step返回的每个游戏自定义的结构化数据转换成StepFrameReq并序列化
        # 转换逻辑由self.game_name决定（场景接入方实现）, 该函数逻辑与game.step返回的数据强相关, 需要接入方仔细实现
        byte_stepframe_req = Parse_StepFrameReq.encode(self.game_name, **{
            'game_id': game_id, 
            'frame_no': frame_no, 
            'frame_state': _frame_state, 
            'terminated': terminated,
            'truncated': truncated,
            'game_info': game_info})
        
        # 发送StepFrameReq并收到StepFrameRsp
        byte_stepframe_rsp = self.__send_stepframereq_recv_stepframersp(byte_stepframe_req)

        if byte_stepframe_rsp == False:
            return game_id, frame_no, _frame_state, terminated, truncated, game_info, True

        # 将StepFrameRsp反序列化得到pb数据，将pb数据转换成能被game.step接受的输入(每个游戏自定义的结构化数据)
        # 转换逻辑由self.game_name决定（场景接入方实现), 该函数逻辑与game.step的参数强相关, 需要接入方仔细实现
        game_id, frame_no, command, stop_game = Parse_StepFrameRsp.decode(self.game_name, byte_stepframe_rsp)

        # 调用真实的游戏step, 推动一帧
        game_id, frame_no, _frame_state, terminated, truncated, game_info = self.env.step(game_id, frame_no, command, stop_game)

        # 额外返回stop_game, aisvr可能发送游戏结束信号
        return game_id, frame_no, _frame_state, terminated, truncated, game_info, stop_game

    def run_once(self):
        game_id, frame_no, _frame_state, terminated, truncated, game_info = self.reset()
        while True:
            game_id, frame_no, _frame_state, terminated, truncated, game_info, stop_game = self.step(
                game_id, frame_no, _frame_state, terminated, truncated, game_info 
            )
            # self.step应额外返回aisvr的stop_game,游戏是否结束由自身状态terminated,truncated和stop_game决定
            if terminated or truncated or stop_game:
                # 将结构化数据转换成StepFrameReq并序列化，转换逻辑由self.game_name决定（场景接入方实现）
                byte_stepframe_req = Parse_StepFrameReq.encode(self.game_name, **{
                    'game_id': game_id, 
                    'frame_no': frame_no, 
                    'frame_state': _frame_state, 
                    'terminated': terminated,
                    'truncated': truncated,
                    'game_info': game_info})
                # 发送StepFrameReq并收到StepFrameRsp, 向worker发送stop_sess
                byte_stepframe_rsp = self.client.stop_sess(byte_stepframe_req)
                break

