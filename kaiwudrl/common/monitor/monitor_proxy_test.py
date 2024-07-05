#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file monitor_proxy_test.py
# @brief
# @author kaiwu
# @date 2023-11-28


import unittest
import json
from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.monitor.monitor_proxy import MonitorProxy


class MonitorProxyTest(unittest.TestCase):
    def test_all(self):
        pass

    # 测试修改conf配置文件里的值
    def test_learner_actor_address(self):
        monitor_proxy = MonitorProxy(None)


if __name__ == "__main__":
    unittest.main()
