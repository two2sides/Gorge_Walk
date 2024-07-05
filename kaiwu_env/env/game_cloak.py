
from kaiwu_env.env.protocol import Parse_StepFrameRsp, Parse_StepFrameReq
from kaiwu_env.env.protocol import Parse_AIServerRequest, Parse_AIServerResponse
from kaiwu_env.env.protocol import SkylarenaDataHandler
from kaiwu_env.utils.common_func import get_uuid
from kaiwu_env.conf import yaml_arena
from kaiwu_env.utils.strategy import strategy_selector


class GameCloak:
    def __init__(self, env, game_name, scene_name='default_scene', logger=None, monitor=None) -> None:
        self.game_name = game_name
        self.env = strategy_selector("scene_wrapper", yaml_arena.gamecore_type, env)
        self.data_handler = SkylarenaDataHandler(self.game_name, logger, monitor)
        # self.F = self.env.F

    def reset(self, usr_conf):

        game_id = get_uuid()
        game_id = yaml_arena.eval_game_id if yaml_arena.train_or_eval == "eval" else game_id
        
        # env.reset拿到游戏的信息
        game_id, frame_no, _frame_state, terminated, truncated, game_info = self.env.reset(game_id, usr_conf)
        
        # reset data_handler
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

        # 将StepFrameRsp反序列化得到pb数据，将pb数据转换成能被game.step接受的输入(每个游戏自定义的结构化数据)
        # 转换逻辑由self.game_name决定（场景接入方实现), 该函数逻辑与game.step的参数强相关, 需要接入方仔细实现
        game_id, frame_no, command, stop_game = Parse_StepFrameRsp.decode(self.game_name, byte_stepframe_rsp)

        # 调用真实的游戏step, 推动一帧, 拿到游戏帧的信息
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

        # 将AIServerRequest反序列化后, 转换成用户调用env.reset或env.step期望获得的结构化数据
        # 转换逻辑由self.game_name决定（场景接入方实现）, 该函数逻辑与env.step返回的数据强相关, 需要接入方仔细实现
        self.game_id, self.frame_no, self.obs, self.score_info, self.terminated, self.truncated, self.env_info = Parse_AIServerRequest.decode(self.game_name, byte_aisvr_req)

        self.data_handler.step(pb_stepframe_req, pb_aisvr_rsp)

        return self.frame_no, self.obs, self.score_info, self.terminated, self.truncated, self.env_info

