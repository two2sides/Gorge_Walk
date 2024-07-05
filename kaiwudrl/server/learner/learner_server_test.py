#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file learner_server_test.py
# @brief
# @author kaiwu
# @date 2023-11-28


import unittest
from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.config.algo_conf import AlgoConf
from kaiwudrl.common.config.app_conf import AppConf
from kaiwudrl.common.utils.cmd_argparser import cmd_args_parse
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.server.learner.learner_server import LearnerServer


class LearnerServerTest(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def proc_flags(configure_file):
        CONFIG.set_configure_file(configure_file)
        CONFIG.parse_learner_configure()

        # 加载配置文件conf/algo_conf.json
        AlgoConf.load_conf(CONFIG.algo_conf)

        # 加载配置文件conf/app_conf.json
        AppConf.load_conf(CONFIG.app_conf)

    def test_all(self):
        # 步骤1, 按照命令行来解析参数
        args = cmd_args_parse(KaiwuDRLDefine.SERVER_LEARNER)
        # 步骤2, 解析参数, 包括业务级别和算法级别
        self.proc_flags(args.conf)
        a = LearnerServer()
        a.start()
        while True:
            pass


if __name__ == "__main__":
    unittest.main()
