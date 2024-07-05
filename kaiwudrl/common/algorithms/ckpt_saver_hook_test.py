#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file ckpt_saver_hook_test.py
# @brief
# @author kaiwu
# @date 2023-11-28


import unittest
from kaiwudrl.common.utils.tf_utils import *
from kaiwudrl.common.algorithms.ckpt_saver_hook import CkptSaverListener


class CkptSaverHookTest(unittest.TestCase):
    def test_save_model(self):
        tensor_x = tf.convert_to_tensor(1)
        listener = CkptSaverListener(inputs={"x": tensor_x}, outputs={"y": tf.identity(tensor_x)})
        ckpt_saver_hook = tf.train.CheckpointSaverHook(
            checkpoint_dir="/tmp",
            save_secs=30,
            listeners=[
                listener,
            ],
        )


if __name__ == "__main__":
    unittest.main()
