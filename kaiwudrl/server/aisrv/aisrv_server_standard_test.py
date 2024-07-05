#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file aisrv_server_standard_test.py
# @brief
# @author kaiwu
# @date 2023-11-28


import unittest
from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.server.aisrv.aisrv_server_standard import AiServer


class AiServerTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_all(self):
        server = AiServer()
        server.run()


if __name__ == "__main__":
    unittest.main()
