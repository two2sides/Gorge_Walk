from kaiwu_env.conf import yaml_arena
if yaml_arena.gamecore_type in ["SGWrapper","RPCWrapper"] and yaml_arena.run_mode != "proxy":
    from sgwrapper import UsrConf
if yaml_arena.gamecore_type == "RPCWrapper":
    import socket
    import pickle
    import struct
    from kaiwu_env.back_to_the_realm.server2game_pb2 import *
from kaiwu_env.conf import yaml_back_to_the_realm_game as game_conf


class BaseEnv:
    def reset(self):
        raise NotImplementedError

    def step(self, act):
        raise NotImplementedError


class PYSceneWrapper:
    def __init__(self, env) -> None:
        self.env = env

    def reset(self, game_id, usr_conf):
        (
            game_id,
            frame_no,
            _frame_state,
            terminated,
            truncated,
            game_info,
        ) = self.env.reset(game_id, usr_conf)
        return game_id, frame_no, _frame_state, terminated, truncated, game_info

    def step(self, game_id, frame_no, command, stop_game):
        (
            game_id,
            frame_no,
            _frame_state,
            terminated,
            truncated,
            game_info,
        ) = self.env.step(game_id, frame_no, command, stop_game)
        return game_id, frame_no, _frame_state, terminated, truncated, game_info

    def __getattr__(self, attr):
        return getattr(self.env, attr)


class SGSceneWrapper:
    def __init__(self, env) -> None:
        self.env = env
        self.env.init()

    def reset(self, game_id, usr_conf):

        from random import sample

        (
            start,
            end,
            treasure_id,
            talent_type,
            treasure_num,
            treasure_random,
        ) = self._read_usr_conf(usr_conf)

        # treasure_random 表示是否随机生成宝箱, 0表示否，1表示是
        # treasure_random 配置文件中默认为0，若同时输入treasure_id与treasure_random == 1优先随机宝箱
        # 从1 - 15中排除掉start和end的位置
        values_to_exclude = [start, end]
        available_treasure_pos = [x for x in range(1, 16) if x not in values_to_exclude]
        if treasure_random == 1:
            treasure_id = sample(available_treasure_pos, treasure_num)

        # print(f"{start}, {end}, {treasure_id},{talent_type}, {treasure_num}")
        usr_conf = UsrConf(start, end, treasure_id, talent_type)

        game_frame = self.env.reset(game_id, usr_conf)

        (
            game_id,
            frame_no,
            _frame_state,
            terminated,
            truncated,
            game_info,
        ) = self.__parse_game_frame(game_frame)

        return game_id, frame_no, _frame_state, terminated, truncated, game_info

    def step(self, game_id, frame_no, command, stop_game):
        game_frame = self.env.step(game_id, frame_no, command, stop_game)
        (
            game_id,
            frame_no,
            _frame_state,
            terminated,
            truncated,
            game_info,
        ) = self.__parse_game_frame(game_frame)

        return game_id, frame_no, _frame_state, terminated, truncated, game_info

    def __parse_game_frame(self, game_frame):
        game_id, frame_no, _frame_state, terminated, truncated, game_info = (
            game_frame.game_id,
            game_frame.frame_no,
            game_frame.frame_state,
            game_frame.terminated,
            game_frame.truncated,
            game_frame.game_info,
        )
        return game_id, frame_no, _frame_state, terminated, truncated, game_info

    # UsrConf数据转换
    def _read_usr_conf(self, usr_conf):
        def __get_value(key):
            if key in game_conf.diy.keys():
                return game_conf.diy[key]
            elif key in game_conf.keys():
                return game_conf[key]
            else:
                raise KeyError

        if usr_conf:
            game_conf.render_config_from_dict(usr_conf)
        else:
            return (
                game_conf.start,
                game_conf.end,
                game_conf.treasure_id,
                game_conf.treasure_num,
                game_conf.treasure_random,
            )

        start = __get_value("start")
        end = __get_value("end")
        treasure_id = __get_value("treasure_id")
        talent_type = __get_value("talent_type")
        treasure_num = __get_value("treasure_num")
        treasure_random = __get_value("treasure_random")

        return start, end, treasure_id, talent_type, treasure_num, treasure_random


class RPCSceneWrapper:
    def __init__(self, env) -> None:

        if not isinstance(env, dict):
            raise RuntimeError("wrong env_type, please check conf")

        # 采用TCP连接和gamecore通信
        self.host = env["rpc_host"]
        self.port = env["rpc_port"]

        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # connect
        try:
            self.client_socket.connect((self.host, self.port))
        except Exception as e:
            print(f"Failed to connect to server: {e}")

    def reset(self, game_id, usr_conf):

        from random import sample

        (
            start,
            end,
            treasure_id,
            talent_type,
            treasure_num,
            treasure_random,
        ) = self._read_usr_conf(usr_conf)

        # treasure_random 表示是否随机生成宝箱, 0表示否，1表示是
        # treasure_random 配置文件中默认为0，若同时输入treasure_id与treasure_random == 1优先随机宝箱
        # 从1 - 15中排除掉start和end的位置
        values_to_exclude = [start, end]
        available_treasure_pos = [x for x in range(1, 16) if x not in values_to_exclude]
        if treasure_random == 1:
            treasure_id = sample(available_treasure_pos, treasure_num)

        # print(f"{start}, {end}, {treasure_id},{talent_type}, {treasure_num}")
        usr_conf = UsrConf(start, end, treasure_id, talent_type)

        message = {
            "message_type": "reset",
            "game_id": game_id,
            "user_conf_start": start,
            "user_conf_end": end,
            "user_conf_treasure_id": treasure_id,
            "user_conf_talent_type": talent_type,
        }
        message = pickle.dumps(message)
        # 发送信息
        self.send_all(message)
        # 接收信息
        response = self.recv_with_length()
        # 反序列化
        gameframe = GameFrame()
        gameframe.ParseFromString(response)
        return (
            gameframe.game_id,
            gameframe.frame_no,
            gameframe.frame_state,
            gameframe.terminated,
            gameframe.truncated,
            gameframe.game_info,
        )

    def step(self, game_id, frame_no, command, stop_game):

        message = {
            "message_type": "step",
            "game_id": game_id,
            "frame_no": frame_no,
            "command": [
                command.heroid,
                command.move_dir,
                command.talent_type,
                command.move_to_pos_x,
                command.move_to_pos_z,
            ],
            "stop_game": stop_game,
        }
        message = pickle.dumps(message)
        # 发送信息
        self.send_all(message)
        # 接收信息
        response = self.recv_with_length()
        # 反序列化
        gameframe = GameFrame()
        gameframe.ParseFromString(response)

        return (
            gameframe.game_id,
            gameframe.frame_no,
            gameframe.frame_state,
            gameframe.terminated,
            gameframe.truncated,
            gameframe.game_info,
        )

    def __parse_game_frame(self, game_frame):
        game_id, frame_no, _frame_state, terminated, truncated, game_info = (
            game_frame.game_id,
            game_frame.frame_no,
            game_frame.frame_state,
            game_frame.terminated,
            game_frame.truncated,
            game_frame.game_info,
        )
        return game_id, frame_no, _frame_state, terminated, truncated, game_info

    def send_all(self, data):
        # 将数据长度打包成4字节的二进制数据
        data_length = struct.pack("!I", len(data))
        self.client_socket.send(data_length)

        bytes_sent = 0
        while bytes_sent < len(data):
            sent = self.client_socket.send(data[bytes_sent:])
            if sent == 0:
                raise RuntimeError("socket connection broken")
            bytes_sent = bytes_sent + sent

    def recv_with_length(self):
        # 首先接收4字节的数据长度
        data_length = self.client_socket.recv(4)
        if not data_length:
            raise RuntimeError("socket connection broken")

        # 将数据长度解包成整数
        length = struct.unpack("!I", data_length)[0]
        # 根据数据长度接收数据
        data = self.recv_all(length)
        return data

    def recv_all(self, length):
        data = b""
        while len(data) < length:
            more = self.client_socket.recv(length - len(data))
            if not more:
                raise RuntimeError("socket connection broken")
            data += more
        return data

    # UsrConf数据转换
    def _read_usr_conf(self, usr_conf):
        def __get_value(key):
            if key in game_conf.diy.keys():
                return game_conf.diy[key]
            elif key in game_conf.keys():
                return game_conf[key]
            else:
                raise KeyError

        if usr_conf:
            game_conf.render_config_from_dict(usr_conf)
        else:
            return (
                game_conf.start,
                game_conf.end,
                game_conf.treasure_id,
                game_conf.treasure_num,
                game_conf.treasure_random,
            )

        start = __get_value("start")
        end = __get_value("end")
        treasure_id = __get_value("treasure_id")
        talent_type = __get_value("talent_type")
        treasure_num = __get_value("treasure_num")
        treasure_random = __get_value("treasure_random")

        return start, end, treasure_id, talent_type, treasure_num, treasure_random
