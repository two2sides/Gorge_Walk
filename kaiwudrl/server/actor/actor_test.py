#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file actor_test.py
# @brief
# @author kaiwu
# @date 2023-11-28


import unittest

try:
    import _pickle as pickle
except ImportError as e:
    import pickle

from kaiwudrl.server.actor.actor_server_sync import ActorServerSync
from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.utils.common_func import TimeIt


class ActorServerTest(unittest.TestCase):
    def setUp(self):
        CONFIG.set_configure_file("conf/kaiwudrl/actor.toml")
        CONFIG.parse_actor_configure()

    def test_pickle_data(self):
        data = [0 for i in range(10000)]
        with TimeIt() as it:
            data = pickle.dumps(data)
        print(it.interval)

        with TimeIt() as it:
            data = pickle.loads(data, encoding="bytes")
        print(it.interval)


if __name__ == "__main__":
    unittest.main()
