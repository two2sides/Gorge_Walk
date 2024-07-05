from kaiwu_env.utils.strategy import strategy_selector 
import sys

class Parse_AIServerRequest:
    # AIServerRequest的encode应该由场景接入方在SkaylarenaDataHandler.StepFrameReq2AISvrReq中实现
    
    @staticmethod
    def decode(game_name, *args, **kargs):
        """
            将AIServerRequest反序列化后, 转换成用户调用env.reset或env.step期望获得的结构化数据
            转换逻辑由self.game_name决定（场景接入方实现）, 该函数逻辑与env.step返回的数据强相关, 需要接入方仔细实现
        """
        return strategy_selector(f"{__name__}.{__class__.__name__}.{sys._getframe().f_code.co_name}", game_name, *args, **kargs)

class Parse_AIServerResponse:
    @staticmethod
    def encode(game_name, *args, **kargs):
        """
            用户env.step传入int或float类型的动作，转换成AIServerResponse并序列化, 
            转换逻辑由self.game_name决定（场景接入方实现）, 该函数逻辑与env.step返回的参数强相关, 需要接入方仔细实现
        """
        return strategy_selector(f"{__name__}.{__class__.__name__}.{sys._getframe().f_code.co_name}", game_name, *args, **kargs)
    
    @staticmethod
    def decode(game_name, *args, **kargs):
        """
            Skaylarena中调用, 用来解析aisvr传递过来的byte, 转换成pb, 一般业务方写成
            return AIServerResponse().ParseFromString(byte_aisvr_rsp) 
        """
        return strategy_selector(f"{__name__}.{__class__.__name__}.{sys._getframe().f_code.co_name}", game_name, *args, **kargs)

class Parse_StepFrameReq:
    @staticmethod
    def encode(game_name, *args, **kargs):
        """
            将game.reset或game.step返回的每个游戏自定义的结构化数据转换成StepFrameReq并序列化
            转换逻辑由self.game_name决定（场景接入方实现）, 该函数逻辑与game.step返回的数据强相关, 需要接入方仔细实现
        """
        return strategy_selector(f"{__name__}.{__class__.__name__}.{sys._getframe().f_code.co_name}", game_name, *args, **kargs)
    
    @staticmethod
    def decode(game_name, *args, **kargs):
        """
            Skaylarena中调用, 用来解析game传递过来的byte, 转换成pb, 一般业务方写成
            return StepFrameReq().ParseFromString(byte_stepframe_req)
        """
        return strategy_selector(f"{__name__}.{__class__.__name__}.{sys._getframe().f_code.co_name}", game_name, *args, **kargs)

class Parse_StepFrameRsp:

    # StepFrameRsp的encode应该由场景接入方在SkaylarenaDataHandler.AISvrRsp2StepFrameRsp中实现
    
    @staticmethod
    def decode(game_name, *args, **kargs):
        """
            将StepFrameRsp反序列化得到pb数据，将pb数据转换成能被game.step接受的输入(每个游戏自定义的结构化数据)
            转换逻辑由self.game_name决定（场景接入方实现), 该函数逻辑与game.step的参数强相关, 需要接入方仔细实现
        """
        return strategy_selector(f"{__name__}.{__class__.__name__}.{sys._getframe().f_code.co_name}", game_name, *args, **kargs)


class BaseSkylarenaDataHandler:
    def reset(self, usr_conf):
        raise NotImplementedError
    
    def step(self, pb_stepframe_req, pb_aisvr_rsp):
        raise NotImplementedError

    def finish(self):
        raise NotImplementedError
    
    def StepFrameReq2AISvrReq(self, pb_stepframe_req):
        raise NotImplementedError

    def AISvrRsp2StepFrameRsp(self, pb_aisvr_rsp):
        raise NotImplementedError

def SkylarenaDataHandler(game_name, *args, **kargs):
    return strategy_selector(f"{__name__}.{sys._getframe().f_code.co_name}", game_name, *args, **kargs)

