#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file cmd_argparser_test.py
# @brief
# @author kaiwu
# @date 2023-11-28


import sys
import unittest
from unittest.mock import patch
from kaiwudrl.common.utils.cmd_argparser import cmd_args_parse


class CmdArgParserTest(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_aisrv_args(self):
        testargs = ["proc", "--actor_adress", "0.0.0.0:8000"]

        with patch.object(sys, "argv", testargs):
            args = cmd_args_parse("aisrv")
            print(args.actor_adress)


if __name__ == "__main__":
    unittest.main()
