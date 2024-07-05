#!/usr/bin/env python
# -*- coding: utf-8 -*-


# @file sample_server_test.py
# @brief
# @author kaiwu
# @date 2023-11-28


import unittest
from kaiwudrl.common.config.config_control import CONFIG


class MsgEngineTest(unittest.TestCase):
    def test_all(self):

        # 解析配置
        CONFIG.set_configure_file("/data/projects/kaiwu-fwk/conf/kaiwudrl/aisrv.toml")
        CONFIG.parse_aisrv_configure()


if __name__ == "__main__":
    unittest.main()
