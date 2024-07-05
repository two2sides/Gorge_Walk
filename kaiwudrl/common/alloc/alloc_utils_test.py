#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file alloc_utils_test.py
# @brief
# @author kaiwu
# @date 2023-11-28


import unittest
import os
from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.alloc.alloc_utils import SERVER_ROLE_CONFIGURE, AllocUtils


class AllocUtilTest(unittest.TestCase):
    def setUp(self) -> None:
        CONFIG.set_configure_file("/data/projects/kaiwu-fwk/conf/kaiwudrl/learner.toml")
        CONFIG.parse_learner_configure()

    def test_registry(self):
        allocUtils = AllocUtils(None)
        print(allocUtils.registry())

    def test_get(self):
        allocUtils = AllocUtils(None)
        print(allocUtils.get(SERVER_ROLE_CONFIGURE["actor"]))
        print(allocUtils.get(SERVER_ROLE_CONFIGURE["learner"]))

    def test_get_self_play(self):
        allocUtils = AllocUtils(None)
        print(allocUtils.get(SERVER_ROLE_CONFIGURE["actor"]), "set100")
        print(allocUtils.get(SERVER_ROLE_CONFIGURE["learner"]), "set100")


if __name__ == "__main__":
    unittest.main()
