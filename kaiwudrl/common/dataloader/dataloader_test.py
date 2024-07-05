#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file dataloader_test.py
# @brief
# @author kaiwu
# @date 2023-11-28


import unittest
import os
from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.dataloader.dataloader import DataLoader


class DataLoaderTest(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_all(self):
        dataloader = DataLoader()


if __name__ == "__main__":
    unittest.main()
