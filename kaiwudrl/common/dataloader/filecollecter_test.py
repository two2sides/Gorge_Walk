#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file filecollecter_test.py
# @brief
# @author kaiwu
# @date 2023-11-28


import unittest
import os
from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.dataloader.filecollecter import FileCollecter


class FileCollecterTest(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_all(self):
        fillcollecter = FileCollecter()


if __name__ == "__main__":
    unittest.main()
