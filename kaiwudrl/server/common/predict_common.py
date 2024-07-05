#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# @file predict_common.py
# @brief
# @author kaiwu
# @date 2023-11-28

import os
import time
import traceback
import numpy as np
from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine

if (
    KaiwuDRLDefine.MODEL_TENSORFLOW_SIMPLE == CONFIG.use_which_deep_learning_framework
    or KaiwuDRLDefine.MODEL_TENSORFLOW_COMPLEX == CONFIG.use_which_deep_learning_framework
):
    from kaiwudrl.common.utils.tf_utils import *

elif KaiwuDRLDefine.MODEL_PYTORCH == CONFIG.use_which_deep_learning_framework:
    from kaiwudrl.common.utils.torch_utils import *
else:
    pass
import random

from kaiwudrl.common.utils.common_func import TimeIt

if CONFIG.aisrv_actor_communication_way == KaiwuDRLDefine.COMMUNICATION_WAY_ZMQ_OPS:
    from kaiwudrl.common.pybind11.zmq_ops.zmq_ops import dump_arrays
from kaiwudrl.common.logging.kaiwu_logger import g_not_server_label
from kaiwudrl.common.utils.common_func import (
    TimeIt,
)

from kaiwudrl.common.checkpoint.model_file_common import (
    get_checkpoint_id_by_re,
)

# 按照需要加载dill
if CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_STANDARD:
    import dill


class PredictCommon(object):
    """
    该类主要是预测的公共类, 因为存在有actor, actor_proxy_local的2个进程使用, 故将代码单独提出公共的, 只是维护一份即可
    """

    def __init__(self, logger) -> None:

        # 下面是因为需要在使用时用到的变量, 故该类里只是定义, 由调用者进行赋值

        # policy和model_wrapper对象的map, 为了支持多agent
        self.policy_model_wrapper_maps = None
        self.policy_conf = None
        self.current_sync_model_version_from_learner = -1
        self.model_file_sync_wrapper = None
        self.last_load_model_aisrv_ip = None
        self.last_load_model_time = 0
        self.logger = logger

        # 统计使用
        self.actor_batch_predict_cost_time_ms = 0
        self.actor_load_last_model_succ_cnt = 0
        self.actor_load_last_model_error_cnt = 0
        self.actor_load_last_model_cost_ms = 0

    def set_actor_load_last_model_cost_ms(self, actor_load_last_model_cost_ms):
        self.actor_load_last_model_cost_ms = actor_load_last_model_cost_ms

    def get_actor_load_last_model_cost_ms(self):
        return self.actor_load_last_model_cost_ms

    def get_actor_load_last_model_succ_cnt(self):
        return self.actor_load_last_model_succ_cnt

    def get_actor_load_last_model_error_cnt(self):
        return self.actor_load_last_model_error_cnt

    def get_actor_batch_predict_cost_time_ms(self):
        return self.actor_batch_predict_cost_time_ms

    def set_actor_batch_predict_cost_time_ms(self, actor_batch_predict_cost_time_ms):
        self.actor_batch_predict_cost_time_ms = actor_batch_predict_cost_time_ms

    def set_policy_model_wrapper_maps(self, policy_model_wrapper_maps):
        self.policy_model_wrapper_maps = policy_model_wrapper_maps

    def set_model_file_sync_wrapper(self, model_file_sync_wrapper):
        self.model_file_sync_wrapper = model_file_sync_wrapper

    def set_policy_conf(self, policy_conf):
        self.policy_conf = policy_conf

    def set_current_sync_model_version_from_learner(self, current_sync_model_version_from_learner):
        self.current_sync_model_version_from_learner = current_sync_model_version_from_learner

    def predict_detail(self, model_wrapper, predict_data, batch_size):
        """
        遵循这样的规则:
        1. 如果是训练, 调用predict函数
        2. 如果是评估, 调用exploit函数
        """
        if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODEL_TRAIN:
            return model_wrapper.predict(predict_data, batch_size)
        elif CONFIG.run_mode == KaiwuDRLDefine.RUN_MODEL_EVAL:
            return model_wrapper.exploit(predict_data)
        else:
            return None

    # actor上的预测predict主函数, 使用框架的predict
    def predict_tensorflow(self):
        size = 0
        pred = None

        batch_size = 1

        try:
            extra_tensors = {
                KaiwuDRLDefine.CLIENT_ID_TENSOR: self.dequeue_tensors[-2],
                KaiwuDRLDefine.COMPOSE_ID_TENSOR: self.dequeue_tensors[-1],
            }

            # 获取policy对应的model_wrapper处理, 其中policy是通过请求里传输的, 找不到则默认第一个model_wrapper
            policy = CONFIG.policy_name
            model_wrapper = self.policy_model_wrapper_maps.get(
                policy, next(iter(self.policy_model_wrapper_maps.values()))
            )

            pred = model_wrapper.predict(extra_tensors, batch_size)
            size = next(iter(pred.values())).shape[0]

            pred["s"] = np.array([self.global_step] * size)

        except Exception as e:
            self.logger.error(
                f"predict failed to run predictor, as {e}, traceback.print_exc() is {traceback.format_exc()}"
            )

        # self.logger.debug(f'after predict, size is {size}, pred is {pred}')

        return size, pred

    # actor上的预测predict主函数, 使用TensorRT
    def predict_tensorrt(self, datas):
        batch_size = len(datas)

        # 数据整理
        state_dict = {}
        state_space = self.policy_conf[CONFIG.policy_name].state.state_space()
        for i, key in enumerate(state_space.keys()):
            state_dict[key] = [datas[j][i].flatten() for j in range(batch_size)]

        res_msgs = []
        for i in range(batch_size):
            # 如果是on-policy则返回actor预测用到的model版本号, data[i][-1]格式形如[[ 0  0 28 1]]
            if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
                datas[i][-1][0][3] = self.current_sync_model_version_from_learner

            res_msgs.append(
                {
                    KaiwuDRLDefine.CLIENT_ID_TENSOR: datas[i][-2],
                    KaiwuDRLDefine.COMPOSE_ID_TENSOR: datas[i][-1],
                }
            )

        sizes = []
        try:
            # 获取policy对应的model_wrapper处理, 其中policy是通过请求里传输的, 找不到则默认第一个model_wrapper
            policy = CONFIG.policy_name
            model_wrapper = self.policy_model_wrapper_maps.get(
                policy, next(iter(self.policy_model_wrapper_maps.values()))
            )

            pred = self.predict_detail(model_wrapper, state_dict, batch_size)
            if pred:
                for i in range(batch_size):
                    res_msgs[i]["pred"] = [p[i] for p in pred]
                    sizes.append(len(pred))

        except Exception as e:
            self.logger.error(
                f"predict failed to run predictor, as {e}, traceback.print_exc() is {traceback.format_exc()}"
            )

        # self.logger.debug(f'after predict, size is {sizes}')

        return sizes, res_msgs

    def standard_load_model_choose_one(self, policy, id):
        """
        根据ID来获取对应的model_file
        """
        current_available_model_files = (
            self.model_file_sync_wrapper.ckpt_sync_warper.get_current_available_model_files()
        )
        if not current_available_model_files:
            self.logger.error(f"current_available_model_files is empty, policy is {policy}, id is {id}")
            return None

        if id == KaiwuDRLDefine.ID_LATEST or id == KaiwuDRLDefine.ID_RANDOM:
            # 因为multiprocessing.Queue没有len属性, 故需要将数据转移到list里
            model_files = []
            while not current_available_model_files.empty():
                model_files.append(current_available_model_files.get())

            if not model_files:
                return None

            if id == KaiwuDRLDefine.ID_LATEST:
                model_file = model_files[-1]
            else:
                model_file = random.choice(model_files)

            # 因为会有多个来进行加载model文件, 如果这里某个进程加载完成了则会清空该队列导致其他进程无法加载, 故这里还需要放回去
            for file in model_files:
                current_available_model_files.put(file)

            return model_file

        else:
            return None

    def standard_load_model_detail(self, policy, id):
        """
        加载model文件的操作如下:
        1. 根据policy找到对应的model_wrapper
        2. 根据id来处理
        2.1 如果是latest, random, 则在FIFO队列里加载
        2.2 如果是具体的数字则去init_model_path加载
        """
        if not policy or not id:
            self.logger.error(f"param is error, policy {policy}, id {id}")
            return False

        # 如果还没有从modelpool拉取下来文件则返回
        if not self.model_file_sync_wrapper.ckpt_sync_warper.get_had_pull_the_first_model_file_success_value():
            self.logger.info(f"had not get pull model file, so return")
            return False

        model_wrapper = self.policy_model_wrapper_maps.get(policy, None)
        if not model_wrapper:
            self.logger.error(f"model_wrapper is None, policy is {policy}")
            return False

        if id == KaiwuDRLDefine.ID_LATEST or id == KaiwuDRLDefine.ID_RANDOM:
            load_model_file = self.standard_load_model_choose_one(policy, id)
            if not load_model_file:
                self.logger.error(f"load_model_file is empty, policy is {policy}, id is {id}")
                return False

            # 需要提取path, 获取id
            path = os.path.dirname(load_model_file)
            id = get_checkpoint_id_by_re(load_model_file)
        else:
            path = f"{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}/init_models"

        with TimeIt() as ti:
            success = model_wrapper.standard_load_model(path, id)

        if self.actor_load_last_model_cost_ms < ti.interval * 1000:
            self.actor_load_last_model_cost_ms = ti.interval * 1000

        if success:
            self.logger.info(f"standard_load_model success, path {path}, id {id}")
            self.actor_load_last_model_succ_cnt += 1
        else:
            self.logger.error(f"standard_load_model failed, path {path}, id {id}")
            self.actor_load_last_model_error_cnt += 1

        return success

    def standard_load_model(
        self,
        policy,
        id,
        ip,
    ):
        """
        1. 针对不同的aisrv来让同一个actor执行load_model的操作
        1.1 在时间间隔内如果获取到第一个即执行
        1.2 在最大时间间隔内如果1.1的aisrv持续的进行load_model时则直接执行, 否则转1.3
        1.3 重新接收第一个需要执行的aisrv, 然后更新时间
        """
        if not policy or not id or not ip:
            logger.error(f"param is error, policy {policy}, id {id}, ip {ip}")
            return

        now = time.time()

        # 如果没有上一次的加载模型请求，或者上一次的请求来自同一个ip，执行加载模型
        if not self.last_load_model_aisrv_ip or self.last_load_model_aisrv_ip == ip:
            if self.standard_load_model_detail(policy, id):
                self.last_load_model_aisrv_ip = ip
                self.last_load_model_time = now
        else:
            # 如果上一次的请求来自不同的ip，但已经超过了最大等待时间，也执行加载模型
            if now - self.last_load_model_time >= CONFIG.choose_aisrv_to_load_model_or_save_model_max_time_seconds:
                if self.standard_load_model_detail(policy, id):
                    self.last_load_model_aisrv_ip = ip
                    self.last_load_model_time = now

            # 否则，不执行加载模型，即忽略来自不同ip的请求，如果它们在最大等待时间内

    def standard_predict_simple(self, datas):
        """
        actor上的预测predict主函数, 使用业务的predict
        """

        # 组装batch, 目前由于是采用多policy和agent的, 故每个包就去预测的, 后期优化成批处理再看下这里
        batch_size = 1

        # 数据整理
        state_dict = {}

        res_msgs = []
        sizes = []

        try:
            for data in datas:
                """
                根据来进行不同的操作:
                1. 数据流, 进行预测
                2. 管理流, 加载model文件
                """
                raw_data = data.get("data")
                client_id = data.get("client_id")
                compose_id = data.get("compose_id")

                message_type = raw_data.get(KaiwuDRLDefine.MESSAGE_TYPE)
                message_value = raw_data.get(KaiwuDRLDefine.MESSAGE_VALUE)

                if message_type == KaiwuDRLDefine.MESSAGE_PREDICT:
                    state_dict = dill.loads(message_value)
                    # 如果是on-policy则返回actor预测用到的model版本号, data[i][-1]格式形如[[ 0  0 28 1]]
                    if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
                        compose_id[0][3] = self.current_sync_model_version_from_learner

                    # 获取policy对应的model_wrapper处理, 其中policy是通过请求里传输的, 找不到则默认第一个model_wrapper
                    policy = compose_id[0][4]
                    model_wrapper = self.policy_model_wrapper_maps.get(
                        policy, next(iter(self.policy_model_wrapper_maps.values()))
                    )

                    # 这里打开DEBUG日志会导致性能下降, 故默认不打开任何级别的日志, 如果需要跟进问题手工修改注释再运行
                    pred = self.predict_detail(model_wrapper, state_dict, batch_size)
                    if pred:
                        result = {
                            "pred": pred,
                            KaiwuDRLDefine.CLIENT_ID_TENSOR: client_id,
                            KaiwuDRLDefine.COMPOSE_ID_TENSOR: compose_id,
                        }
                        res_msgs.append(result)
                        sizes.append(len(pred))

                elif message_type == KaiwuDRLDefine.MESSAGE_LOAD_MODEL:
                    self.standard_load_model(
                        message_value.get("policy"),
                        message_value.get("id"),
                        message_value.get("ip"),
                    )

                else:
                    self.logger.error(
                        f"actor_proxy_local un support message_type is {message_type}",
                        g_not_server_label,
                    )

        except Exception as e:
            self.logger.error(
                f"actor_proxy_local failed to run predictor, as {e}, "
                f"traceback.print_exc() is {traceback.format_exc()}",
                g_not_server_label,
            )

        return sizes, res_msgs

    def predict_simple(
        self,
        datas,
    ):
        """
        actor上的预测predict主函数, 使用业务的predict
        """

        # 组装batch, 目前由于是采用多policy和agent的, 故每个包就去预测的, 后期优化成批处理再看下这里
        batch_size = 1

        # 数据整理
        state_dict = {}

        res_msgs = []
        sizes = []

        # 默认按照第一个policy的state
        state_space = self.policy_conf[CONFIG.policy_name].state.state_space()

        try:
            for data in datas:
                raw_data = data.get("data")
                client_id = data.get("client_id")
                compose_id = data.get("compose_id")

                for j, key in enumerate(state_space.keys()):
                    state_dict[key] = [raw_data[j].flatten()]

                # 如果是on-policy则返回actor预测用到的model版本号, data[i][-1]格式形如[[ 0  0 28 1]]
                if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
                    compose_id[0][3] = self.current_sync_model_version_from_learner

                # 获取policy对应的model_wrapper处理, 其中policy是通过请求里传输的, 找不到则默认第一个model_wrapper
                policy = compose_id[0][4]
                model_wrapper = self.policy_model_wrapper_maps.get(
                    policy, next(iter(self.policy_model_wrapper_maps.values()))
                )
                # 这里打开DEBUG日志会导致性能下降, 故默认不打开任何级别的日志, 如果需要跟进问题手工修改注释再运行
                pred = self.predict_detail(model_wrapper, state_dict, batch_size)
                if pred:
                    for i in range(batch_size):
                        result = {
                            "pred": [p[i] for p in pred],
                            KaiwuDRLDefine.CLIENT_ID_TENSOR: client_id,
                            KaiwuDRLDefine.COMPOSE_ID_TENSOR: compose_id,
                        }
                        res_msgs.append(result)
                        sizes.append(len([p[i] for p in pred]))

        except Exception as e:
            self.logger.error(
                f"actor_proxy_local failed to run  predictor, as {e}, "
                f"traceback.print_exc() is {traceback.format_exc()}",
                g_not_server_label,
            )

        # self.logger.debug(f'after actor_proxy_local, size is {sizes}')

        return sizes, res_msgs

    def predict(self, datas):
        """
        预测函数
        """

        # 返回的数据格式
        size = 0
        pred = None

        if not datas:
            return size, pred

        if CONFIG.distributed_tracing:
            self.logger.info(
                f"actor_proxy_local distributed_tracing predict start",
                g_not_server_label,
            )

        if KaiwuDRLDefine.MODEL_TENSORFLOW_SIMPLE == CONFIG.use_which_deep_learning_framework:
            with TimeIt() as ti:
                if CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_NORMAL:
                    size, pred = self.predict_simple(datas)
                elif CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_STANDARD:
                    size, pred = self.standard_predict_simple(datas)
                else:
                    pass

            # tensorflow的运行机制, TensorFlow 首先会构建计算图（Computation Graph)，
            # 这是一个表示计算操作和数据流的图结构。构建计算图需要一些额外的时间，因此第一次执行 session.run() 时会比较耗时。
            # 作为统计, 因为该值是动态变化的, 故第一次可能比较高, 不影响后续统计
            if self.actor_batch_predict_cost_time_ms < ti.interval * 1000:
                self.actor_batch_predict_cost_time_ms = ti.interval * 1000

        elif KaiwuDRLDefine.MODEL_TENSORFLOW_COMPLEX == CONFIG.use_which_deep_learning_framework:
            with TimeIt() as ti:
                size, pred = self.predict_tensorflow()

            # tensorflow的运行机制, TensorFlow 首先会构建计算图（Computation Graph)，
            # 这是一个表示计算操作和数据流的图结构。构建计算图需要一些额外的时间，因此第一次执行 session.run() 时会比较耗时。
            # 作为统计, 因为该值是动态变化的, 故第一次可能比较高, 不影响后续统计
            if self.actor_batch_predict_cost_time_ms < ti.interval * 1000:
                self.actor_batch_predict_cost_time_ms = ti.interval * 1000

        elif KaiwuDRLDefine.MODEL_PYTORCH == CONFIG.use_which_deep_learning_framework:
            with TimeIt() as ti:
                if CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_NORMAL:
                    size, pred = self.predict_simple(datas)
                elif CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_STANDARD:
                    size, pred = self.standard_predict_simple(datas)
                else:
                    pass

            if self.actor_batch_predict_cost_time_ms < ti.interval * 1000:
                self.actor_batch_predict_cost_time_ms = ti.interval * 1000

        elif KaiwuDRLDefine.MODEL_TCNN == CONFIG.use_which_deep_learning_framework:
            pass

        elif KaiwuDRLDefine.MODEL_TENSORRT == CONFIG.use_which_deep_learning_framework:
            with TimeIt() as ti:
                size, pred = self.predict_tensorrt(datas)

            if self.actor_batch_predict_cost_time_ms < ti.interval * 1000:
                self.actor_batch_predict_cost_time_ms = ti.interval * 1000

        else:
            self.logger.error(
                f"actor_proxy_local error use_which_deep_learning_framework "
                f"{CONFIG.use_which_deep_learning_framework}, "
                f"only support {KaiwuDRLDefine.MODEL_TCNN}, {KaiwuDRLDefine.MODEL_PYTORCH}, "
                f"{KaiwuDRLDefine.MODEL_TENSORFLOW_COMPLEX}, {KaiwuDRLDefine.MODEL_TENSORFLOW_SIMPLE}",
                g_not_server_label,
            )

            return size, pred

        if CONFIG.distributed_tracing:
            self.logger.info(f"actor_proxy_local distributed_tracing predict end", g_not_server_label)

        # 处理actor --> aisrv的回包
        if CONFIG.distributed_tracing:
            self.logger.info(
                f"actor_proxy_local distributed_tracing predict put actor_server predict result start",
                g_not_server_label,
            )

        if CONFIG.distributed_tracing:
            self.logger.info(
                f"actor_proxy_local distributed_tracing predict put actor_server predict result end",
                g_not_server_label,
            )

        return size, pred
