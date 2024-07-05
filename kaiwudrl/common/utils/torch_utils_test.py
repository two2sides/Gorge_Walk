#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file torch_utils_test.py
# @brief
# @author kaiwu
# @date 2023-11-28


import torch
import unittest
from kaiwudrl.common.utils.torch_utils import torch_is_gpu_available, compiled_func_name


class TFUtilsTest(unittest.TestCase):
    def setUp(self) -> None:
        pass

    # 测试torch compile func
    def test_torch_compile_func(self):
        @torch.jit.script
        def add(a, b):
            return a + b

        compiled_func_name = compiled_func_name(add)
        print(compiled_func_name(1, 2))


if __name__ == "__main__":
    unittest.main()
