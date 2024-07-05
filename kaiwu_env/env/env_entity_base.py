
from kaiwu_env.conf import yaml_arena
from kaiwu_env.zlib.rrrclient import client_for_drl, client_for_arena
import kaiwu_env
from pickle import dumps, loads
from kaiwu_env.env.protocol import Parse_AIServerRequest, Parse_AIServerResponse
from kaiwu_env.utils.strategy import strategy_selector 
import sys
from kaiwu_env.conf import ini_rrrsvr


class StrategySenderSelector:
    @staticmethod
    def kaiwu_env():
        sender = client_for_arena(1, ini_rrrsvr.client.url_ftend)
        sender.send(None)
        return sender
    
    @staticmethod
    def drl():
        return client_for_drl(1)


def run_env_entity(game_name, scene_name='default_scene'):
    env = kaiwu_env.make(game_name)
    sender = strategy_selector(f"{__name__}.{sys._getframe().f_code.co_name}", yaml_arena.aisvr_type)
    while True:
        _obs = env.reset()
        reward, done, info = 0, False, {}
        while True:
            aisvr_req = Parse_AIServerRequest.encode(game_name, **{'observation': _obs, 'score': reward, 'done': done, 'info': info})
            act = sender.send(aisvr_req)
            act = Parse_AIServerResponse.decode(game_name, act)
            _obs, reward, done, info = env.step(act)
            
            if done:
                # 最后一帧需要单独处理
                aisvr_req = Parse_AIServerRequest.encode(game_name, **{'observation': _obs, 'score': reward, 'done': done, 'info': info})
                act = sender.send(aisvr_req)
                break

