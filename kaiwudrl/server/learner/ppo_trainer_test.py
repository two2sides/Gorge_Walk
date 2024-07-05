#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file ppo_trainer_test.py
# @brief
# @author kaiwu
# @date 2023-11-28


import unittest
from kaiwudrl.server.learner.ppo_trainer import PPOTrainer
from kaiwudrl.common.config.config_control import CONFIG


class PPOTrainerTest(unittest.TestCase):
    def setUp(self):
        CONFIG.set_configure_file("/data/projects/kaiwu-fwk/conf/kaiwudrl/learner.toml")
        CONFIG.parse_learner_configure()


if __name__ == "__main__":
    unittest.main()
