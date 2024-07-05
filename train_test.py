#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project :gorge_walk
@File    :train_test.py
@Author  :kaiwu
@Date    :2022/10/20 11:43

"""

import time
import os
import sys
import platform
from multiprocessing import Process
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.common.utils.http_utils import http_utils_request
from kaiwudrl.common.utils.common_func import (
    stop_process_by_name,
    stop_process_by_cmdline,
    python_exec_shell,
    find_pids_by_cmdline,
    scan_for_errors,
)
import kaiwudrl.server.learner.learner as learner
import kaiwudrl.server.aisrv.aisrv as aisrv
from kaiwudrl.common.config.config_control import CONFIG
from typing import List

# To run the train_test, you must modify the algorithm name here.
# It must be one of dynamic_programming, monte_carlo, q_learning, sarsa or diy.
# Simply modify the value of the algorithm_name variable.
# 运行train_test前必须修改这里的算法名字, 必须是dynamic_programming、monte_carlo、q_learning、sarsa、diy里的一个, 修改algorithm_name的值即可
algorithm_name_list = [
    "dynamic_programming",
    "monte_carlo",
    "q_learning",
    "sarsa",
    "diy",
]
algorithm_name = "dynamic_programming"


# train
# 训练
def train():

    start_time = time.time()

    # To stop a running process
    # 停止已经启动的进程
    stop()

    # To modify the value in the environment variable and initiate training for the learner as soon as possible
    # 修改环境变量里的值, 尽快让learner进行训练
    os.environ.update(
        {
            "use_ckpt_sync": "False",
            "replay_buffer_capacity": "10",
            "preload_ratio": "10",
            "train_batch_size": "2",
            "use_prometheus": "True",
            # "prometheus_pushgateway": "localhost:9091",
        }
    )

    # To modify the configuration and execute the script directly
    # 修改配置, 调用脚本直接执行
    if algorithm_name not in algorithm_name_list:
        print("\033[92m" + f"algorithm_name: {algorithm_name} not in list {algorithm_name_list}" + "\033[0m")

        stop()
        sys.exit()

    CONFIG.set_configure_file("conf/kaiwudrl/learner.toml")
    CONFIG.parse_learner_configure()

    python_exec_shell(f"sh /root/tools/change_algorithm_all.sh {algorithm_name}")
    print(f"current algorithm_name is {algorithm_name}")

    # Setting the sample transmission to either Reverb or ZMQ based on the operating system
    # 根据不同的操作系统设置发送样本的是reverb还是zmq
    architecture = platform.machine()
    platform_maps = {
        "aarch64": "zmq",
        "arm64": "zmq",
        "x86_64": "reverb",
        "AMD64": "reverb",
    }
    sample_tool_type = platform_maps.get(architecture)
    if sample_tool_type is None:
        print(f"Architecture '{architecture}' may not exist or not be supported.")
    else:
        result_code, result_str = python_exec_shell(f"sh tools/change_sample_server.sh {sample_tool_type}")
        if result_code != 0:
            raise ValueError(f"Execution error! Please check the error detail: {result_str}")

    # To delete previous model files and logs
    # 删除以前的model文件和日志
    python_exec_shell(f"rm -rf {CONFIG.user_ckpt_dir}/{CONFIG.app}_{algorithm_name}/*")
    python_exec_shell(f"rm -rf {CONFIG.log_dir}/*")

    # To start aisrv, learner, actor, and battlesrv
    # 启动aisrv, learner, actor, battlesrv
    procs: List[Process] = []
    procs.append(Process(target=learner.main, name="learner"))
    procs.append(Process(target=aisrv.main, name="aisrv"))

    for proc in procs:
        proc.start()
        time.sleep(10)
        check(proc)

    while True:

        # For dynamic_programming, the generation of a model file on the learner indicates a successful run,
        # while for other algorithms, the method of obtaining monitoring values is adopted
        # 对于dynamic_programming, learner上有model文件生成即代表本次运行成功, 对于其他算法采用获取监控值的方法
        if algorithm_name == "dynamic_programming":
            success = check_model_file_exist()
        else:
            success = check_train_success_by_monitor()

        if success:
            stop()

            time.sleep(5)
            print(
                "\033[1;31m"
                + f"Train test succeed, will exit, cost {time.time() - start_time:.2f} seconds "
                + "\033[0m"
            )

            sys.exit()

        time.sleep(1)
        for proc in procs:
            check(proc)


# To check if a process is alive, any error log
# 检测进程是否存活, 是否有错误日志
def check(proc: Process):
    if proc.is_alive():

        # If an error log is generated, exit early
        # 如果有错误日志产生, 提前退出
        if scan_for_errors(CONFIG.log_dir, error_indicator="ERROR"):
            stop()

            time.sleep(5)
            print("\033[1;31m" + "find error log, please check" + "\033[0m")
            sys.exit()
    else:
        stop()

        time.sleep(5)
        print("\033[1;31m" + f"{proc.name} is not alive, please check error log" + "\033[0m")
        sys.exit()


# Determine the success of training based on the reported monitoring values.
# If the number of training steps is greater than 0,
# it indicates a successful training
# 按照上报监控的值来判断训练是否成功, 如果训练步数大于0则代表训练成功
def check_train_success_by_monitor():
    try:
        pushgateway = os.environ.get("prometheus_pushgateway")
        url = f"http://{pushgateway}/api/v1/metrics"
        resp = http_utils_request(url)
        if not resp:
            return False

        datas = resp.get("data", [])
        pids = find_pids_by_cmdline("train_test")
        pids = [str(pid) for pid in pids]

        for data in datas:
            if "train_global_step" not in data:
                continue

            train_global_step = data.get("train_global_step", {})
            metrics = train_global_step.get("metrics", [])
            if process_monitor_metrics(metrics, pids):
                return True

        return False

    except Exception:
        return False


# Processing monitor metrics
# 处理监控指标
def process_monitor_metrics(metrics: List, pids: List[str]):
    for metric in metrics:
        labels = metric.get("labels", {})
        value = metric.get("value", 0)

        job = labels.get("job", None)
        if job:
            for pid in pids:
                if pid in job and int(value) > 0:
                    return True

    return False


# To ensure the successful saving of the model file for this run
# 确保本次运行的模型保存文件成功
def check_model_file_exist():
    directory = f"{CONFIG.user_ckpt_dir}/{CONFIG.app}_{algorithm_name}"
    if os.path.isdir(directory):
        id_list_file = f"{directory}/id_list"
        if os.path.exists(id_list_file):
            return True

    return False


# To stop a process
# 停止进程
def stop():
    stop_process_by_name(KaiwuDRLDefine.SERVER_AISRV)
    stop_process_by_name(KaiwuDRLDefine.SERVER_LEARNER)
    stop_process_by_cmdline(KaiwuDRLDefine.TRAIN_TEST_CMDLINE, os.getpid())


if __name__ == "__main__":
    train()
