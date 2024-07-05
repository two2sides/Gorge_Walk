#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file model_file_save_test.py
# @brief
# @author kaiwu
# @date 2023-11-28


import unittest
import tempfile
import json
from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.checkpoint.model_file_save import ModelFileSave
from distutils.dir_util import copy_tree, remove_tree
from kaiwudrl.common.config.algo_conf import AlgoConf
from kaiwudrl.common.config.app_conf import AppConf
from kaiwudrl.common.checkpoint.model_file_common import (
    model_file_signature_verify_by_file,
)
from kaiwudrl.common.utils.common_func import (
    base64_encode,
    base64_decode,
    generate_private_key,
    load_public_key_by_data,
    load_private_key_by_data,
    public_signature_verify_by_data,
    generate_private_signature_by_data,
    python_exec_shell,
    set_env_variable,
)
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.backends import default_backend


class ModelFileSaveTest(unittest.TestCase):
    def setUp(self) -> None:

        # 设置环境变量
        if not set_env_variable("./public_key.pem", "public_key_content"):
            print(f"set_env_variable public_key_content error")
        else:
            print(f"set_env_variable public_key_content success")

        if not set_env_variable("./private_key.pem", "private_key_content"):
            print(f"set_env_variable private_key_content error")
        else:
            print(f"set_env_variable private_key_content success")

        CONFIG.set_configure_file("conf/kaiwudrl/learner.toml")
        CONFIG.parse_learner_configure()

        # 加载配置文件conf/algo_conf.json
        AlgoConf.load_conf(CONFIG.algo_conf)

        # 加载配置文件conf/app_conf.json
        AppConf.load_conf(CONFIG.app_conf)

        print(f"public_key_content is {CONFIG.public_key_content}")

    def test_model_file_save(self):
        pass

        # model_file_save = ModelFileSave()
        # model_file_save.start()

    @unittest.skip("skip this test")
    def test_file(self):
        local_and_remote_dirs = {
            "ckpt_dir": "/data/ckpt",
            "summary_dir": "/data/summary",
            "pb_model": "/data/pb_model",
        }

        temp_remote_dirs = {}
        for _, local_and_remote_dir in local_and_remote_dirs.items():
            print(local_and_remote_dir)
            target_dir = tempfile.mkdtemp()
            print(target_dir)
            copy_tree(local_and_remote_dir, target_dir)
            temp_remote_dirs[target_dir] = target_dir

        print(temp_remote_dirs)

    @unittest.skip("skip this test")
    def test_save_model_file_to_cos(self):
        model_file_save = ModelFileSave()
        model_file_save.save_model_file_to_cos()

    @unittest.skip("skip this test")
    def test_clear_dir(self):
        model_file_save = ModelFileSave()
        model_file_save.clear_dir()

    @unittest.skip("skip this test")
    def test_generate_private_key(self):
        generate_private_key()

    @unittest.skip("skip this test")
    def test_rsa(self):
        data = {
            "created_at": "2024-04-15T15:13:56.381301+00:00",
            "train_time": 46,
            "train_step": 9,
            "platform": "tencent_kaiwu",
            "business": "competition",
            "user_id": "user_id",
            "team_id": "team_id",
            "project_code": "back_to_the_realm",
            "project_version": "1.1.1",
            "task_id": "uuid100",
            "algorithm": "dqn",
            "model_file_name": "back_to_the_realm-uuid100-dqn-9-2024_04_15_23_13_56-1.1.1.zip",
            "model_file_hash": "d124f05b82da9a4ea0a36a6eecb6e867486abd8382d33459db811c0d326fe2c2",
        }

        with open(
            f"/workspace/train/backup_model/back_to_the_realm-uuid100-dqn-9-2024_04_16_16_04_08-1.1.1.json",
            "r",
        ) as f:
            data = json.load(f)
        del data["signature"]

        output = json.dumps(data)
        json_dumps = output.encode()
        print(f"json_dumps is {json_dumps}")

        private_key = load_private_key_by_data(CONFIG.private_key_content)
        private_signature = generate_private_signature_by_data(json_dumps, private_key)
        print(f"private_signature is {private_signature}")
        private_signature_base64_encode = base64_encode(private_signature)
        print(f"private_signature is {private_signature_base64_encode}")
        private_signature_base64_decode = base64_decode(private_signature_base64_encode)

        data["a"] = 1
        del data["a"]
        json_dumps = json.dumps(data).encode()

        public_key = load_private_key_by_data(CONFIG.public_key_content)
        if not public_signature_verify_by_data(public_key, json_dumps, private_signature_base64_decode):
            print(f"public_signature_verify_by_data failed")
        else:
            print(f"public_signature_verify_by_data success")

    @unittest.skip("skip this test")
    def test_model_file_verify(self):
        """
        验证数字签名正确性
        """
        success = model_file_signature_verify_by_file(
            "/workspace/train/backup_model/back_to_the_realm-uuid100-dqn-3839-2024_04_15_14_12_22-1.1.1.zip"
        )
        print(f"model_file_signature_verify_by_file result is {success}")


if __name__ == "__main__":
    unittest.main()
