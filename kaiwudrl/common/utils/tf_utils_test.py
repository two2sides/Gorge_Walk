#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file tf_utils_test.py
# @brief
# @author kaiwu
# @date 2023-11-28


import unittest
from kaiwudrl.common.utils.tf_utils import convert_pbtxt_to_pb, convert_pb_to_pbtxt


class TFUtilsTest(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_pb_to_pbtxt(self):
        # .pbtxt转换为.pb
        pb_txt_file_name = "/data/ckpt/graph.pbtxt"
        target_dir = "/data/ckpt/"
        pb_file_name = "graph.pb"
        convert_pbtxt_to_pb(pb_txt_file_name, target_dir, pb_file_name)
        print("convert_pbtxt_to_pb success")

        # .pb转换为.pbtxt
        pb_file_name = "/data/ckpt/graph.pb"
        pb_txt_file_name = "/data/ckpt/graph_new.pbtxt"
        convert_pb_to_pbtxt(pb_file_name, target_dir, pb_txt_file_name)
        print("convert_pb_to_pbtxt success")


if __name__ == "__main__":
    unittest.main()
