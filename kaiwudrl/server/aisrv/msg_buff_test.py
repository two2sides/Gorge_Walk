#!/usr/bin/env python
# -*- coding: utf-8 -*-


# @file msg_buff_test.py
# @brief
# @author kaiwu
# @date 2023-11-28


import threading
import unittest

from kaiwudrl.common.utils.common_func import Context
from kaiwudrl.server.aisrv.msg_buff import MsgBuff
from kaiwudrl.common.config.config_control import CONFIG


def consumer(msg_buff):
    msg = msg_buff.recv_msg()
    msg_buff.send_msg(msg)


class MsgEngineTest(unittest.TestCase):
    def test_all(self):

        # 解析配置
        CONFIG.set_configure_file("/data/projects/kaiwu-fwk/conf/kaiwudrl/aisrv.toml")
        CONFIG.parse_aisrv_configure()

        context = Context()
        context.slot_id = 1
        context.client_address = "127.0.0.1:8080"

        msg_engine = MsgBuff(context)
        t = threading.Thread(target=consumer, args=(msg_engine,))
        t.start()

        msg = msg_engine.update("Hello Kaiwu!")
        self.assertEqual(msg, "Hello Kaiwu!")

        t.join()


if __name__ == "__main__":
    unittest.main()
