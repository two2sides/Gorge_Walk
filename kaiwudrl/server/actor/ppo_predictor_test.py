#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file ppo_predictor_test.py
# @brief
# @author kaiwu
# @date 2023-11-28


import unittest
from kaiwudrl.server.actor.ppo_predictor import PPOPredictor
from kaiwudrl.common.config.config_control import CONFIG


class PPOPredictorTest(unittest.TestCase):
    def setUp(self):
        CONFIG.set_configure_file("/data/projects/kaiwu-fwk/conf/kaiwudrl/actor.toml")
        CONFIG.parse_actor_configure()


if __name__ == "__main__":
    unittest.main()
