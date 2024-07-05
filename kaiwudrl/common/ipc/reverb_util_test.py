#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file reverb_util_test.py
# @brief
# @author kaiwu
# @date 2023-11-28


import datetime
import unittest
from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.logging.kaiwu_logger import KaiwuLogger
from kaiwudrl.common.ipc.reverb_util import RevervbUtil


class ReverbUtilTest(unittest.TestCase):
    def setUp(self) -> None:
        self.learner_addr = "127.0.0.1"
        self.reverb_util = RevervbUtil(self.learner_addr, None)

    def test_connect(self):
        learner_addr = "127.0.0.1"

    def test_insert_data(self):
        pass


if __name__ == "__main__":
    unittest.main()
