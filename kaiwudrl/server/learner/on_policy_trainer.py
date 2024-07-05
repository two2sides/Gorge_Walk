#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file on_policy_trainer.py
# @brief
# @author kaiwu
# @date 2023-11-28


from multiprocessing import Value
import time
import traceback
import schedule
import os
import threading

# 按照需要导入
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
from kaiwudrl.server.learner.trainer import Trainer
from kaiwudrl.common.replay_buffer.replay_buffer_wrapper import ReplayBufferWrapper
from kaiwudrl.common.checkpoint.model_file_sync_wrapper import ModelFileSyncWrapper
from kaiwudrl.common.checkpoint.model_file_sync import ModelFileSync
from kaiwudrl.common.checkpoint.model_file_save import ModelFileSave
from kaiwudrl.common.utils.common_func import (
    TimeIt,
    set_schedule_event,
    make_single_dir,
    actor_learner_aisrv_count,
    get_host_ip,
    get_uuid,
    register_sigterm_handler,
    stop_process_by_name,
)
from kaiwudrl.common.alloc.alloc_proxy import AllocProxy
from kaiwudrl.common.monitor.monitor_proxy_process import MonitorProxy
from kaiwudrl.common.ipc.zmq_util import ZmqServer, ZmqClient
from kaiwudrl.common.alloc.alloc_utils import AllocUtils
from kaiwudrl.common.config.app_conf import AppConf
from kaiwudrl.common.checkpoint.model_file_common import (
    clear_id_list_file,
    update_id_list,
    check_path_id_valid,
    clear_user_ckpt_dir,
    process_stop_write_file,
)
from kaiwudrl.common.algorithms.model_wrapper_common import (
    create_standard_model_wrapper,
    create_normal_model_wrapper,
)


class OnPolicyTrainer(Trainer):
    @property
    def tensor_names(self):
        raise NotImplementedError

    @property
    def tensor_dtypes(self):
        raise NotImplementedError

    @property
    def tensor_shapes(self):
        raise NotImplementedError

    def __init__(self, name):
        super(OnPolicyTrainer, self).__init__(name)

        self.cached_local_step = -1

        self.local_step = Value("d", -1)

        # policy和model_wrapper对象的map, 为了支持多agent
        self.policy_model_wrapper_maps = {}

        # replay_buffer
        self.replay_buffer_wrapper = ReplayBufferWrapper(
            self.tensor_names, self.tensor_dtypes, self.tensor_shapes, self.logger
        )
        self.replay_buffer_wrapper.init()
        self.replay_buffer_wrapper.extra_threads()

    # 当作为主learner时, 需要保存ckpt文件
    def chief_only_hooks(self):
        with tf.device(f"{self.model_wrapper.learner_device}/cpu:0"):
            return [self.model.ckpt_saver_hook()]

    def train_hooks(self):
        with tf.device(f"{self.model_wrapper.learner_device}/cpu:0"):
            return self.replay_buffer_wrapper.train_hooks(self.model_wrapper.local_step)

    def start_learner_process_by_type(self):
        """
        根据不同的启动方式进行处理:
        1. 正常启动, 无需做任何操作, tensorflow会加载容器里的空的model文件启动
        2. 加载配置文件启动, 需要从COS拉取model文件再启动, tensorflow会加载容器里的model文件启动
        """

        # 按照需要引入ModelFileSave
        self.model_file_saver = ModelFileSave()
        self.model_file_saver.start_actor_process_by_type(self.logger)

    # learner周期性的加载七彩石修改配置, 主要包括进程独有的和公共的
    def rainbow_activate(self):
        self.rainbow_wrapper.rainbow_activate_single_process(KaiwuDRLDefine.SERVER_MAIN, self.logger)
        self.rainbow_wrapper.rainbow_activate_single_process(CONFIG.svr_name, self.logger)

    # learn上的训练train函数流程, 返回是否真实的训练
    def train_detail(self):
        """
        直接调用业务返回的数据格式上报, 框架不关心具体的类型和值, 格式是map
        """
        with TimeIt() as ti:
            (
                app_monitor_data,
                has_model_file_changed,
                model_file_id,
            ) = self.model_wrapper.train()
            if app_monitor_data and isinstance(app_monitor_data, dict):
                self.app_monitor_data = app_monitor_data

        if self.batch_train_cost_time_ms < ti.interval * 1000:
            self.batch_train_cost_time_ms = ti.interval * 1000

        if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY and has_model_file_changed:
            self.current_sync_model_version_from_learner = model_file_id

            if CONFIG.on_policy_by_way == KaiwuDRLDefine.ALGORITHM_ON_POLICY_WAY_TIME_INTERVAL:
                self.learner_on_policy_process(True)

    # aisrv启动的learner的on-policy流程
    def learner_on_policy_process_by_aisrv(self):
        if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
            if (
                CONFIG.on_policy_by_way == KaiwuDRLDefine.ALGORITHM_ON_POLICY_WAY_EPISODE
                or CONFIG.on_policy_by_way == KaiwuDRLDefine.ALGORITHM_ON_POLICY_WAY_STEP
            ):
                self.learner_on_policy_process_by_aisrv_detail()

    # learner周期性访问aisrv, 发送心跳请求, aisrv主动发起来on-policy的流程, 此时需要learner朝aisrv发送自己的client_id
    def on_policy_learner_recv_aisrv_heartbeat_req_resp(self):
        if not self.aisrv_zmq_client_map:
            return

        learner_send_to_aisrv_heartbeat_success_count = 0
        for aisrv_ip, zmq_client in self.aisrv_zmq_client_map.items():
            send_data = {
                KaiwuDRLDefine.MESSAGE_TYPE: KaiwuDRLDefine.ON_POLICY_MESSAGE_HEARTBEAT_REQUEST,
                KaiwuDRLDefine.MESSAGE_VALUE: KaiwuDRLDefine.ON_POLICY_MESSAGE_HEARTBEAT_REQUEST,
            }
            zmq_client.send(send_data, binary=False)
            self.logger.debug(f"train send heartbeat request to aisrv: {aisrv_ip} success")

        """
        同步等待心跳响应回包, 这里因为aisrv在心跳包里会带上是否让learner启动on-policy流程, 故需要注意的点:
        1. 计算心跳回包和计算aisrv启动on-policy的数量需要分开
        2. end_time只能增加1次, 否则进入死循环, 无法退出
        3. 理论上因为aisrv在CONFIG.on_policy_timeout_seconds时间里判断是否有on-policy流程, 故站在leaner的角度看是需要获取所有的aisrv的响应回包
           理论上该值为len(self.aisrv_zmq_client_map) * CONFIG.on_policy_timeout_seconds,
           但是配置过大导致learner的主循环阻塞, 故折中设置为2 * CONFIG.on_policy_timeout_seconds
        """
        end_time = time.time() + 2 * CONFIG.on_policy_timeout_seconds
        success_recv_aisrv_ip = []
        while time.time() < end_time:
            for aisrv_ip, zmq_client in self.aisrv_zmq_client_map.items():
                if aisrv_ip not in success_recv_aisrv_ip:
                    try:
                        recv_data = zmq_client.recv(block=False, binary=False)
                        if recv_data:
                            # 收到了aisrv让learner启动on-policy流程时, 需要发送确认响应
                            if (
                                recv_data[KaiwuDRLDefine.MESSAGE_TYPE]
                                == KaiwuDRLDefine.ON_POLICY_MESSAGE_HEARTBEAT_RESPONSE
                            ):
                                success_recv_aisrv_ip.append(aisrv_ip)

                                learner_send_to_aisrv_heartbeat_success_count += 1
                                self.logger.debug(f"train learner recv aisrv {aisrv_ip}  heartbeat response success")

                            else:
                                pass

                    except Exception as e:
                        # 减少CPU争用
                        time.sleep(CONFIG.idle_sleep_second)

            """
            跳出循环的条件:
            1. 满足所有的aisrv的心跳收到请求则跳出循环
            """
            if learner_send_to_aisrv_heartbeat_success_count == len(self.aisrv_zmq_client_map):
                break

        """
        如果本周期内收到全部的learner发给aisrv的心跳请求响应, 则不进行下一步处理, 说明aisrv、learner都是正常的
        如果本周期内没有全部的learner发给aisrv的心跳请求响应, 则清空self.aisrv_zmq_client_map, 再重新拉取aisrv列表
        """
        if learner_send_to_aisrv_heartbeat_success_count == len(self.aisrv_zmq_client_map):
            self.logger.info(
                f"train learner recv all aisrv heartbeat response success, "
                f"count: {learner_send_to_aisrv_heartbeat_success_count}"
            )

        else:
            self.logger.error(f"train learner not recv all aisrv heartbeat response, retry next time")

            self.aisrv_zmq_client_map.clear()
            self.on_policy_learner_get_aisrv_address()

    # learner接收来自aisrv的on-policy请求
    def on_policy_learner_recv_aisrv_on_policy_req_resp(self):
        if not self.aisrv_zmq_client_map:
            return

        """
        同步等待心跳响应回包, 这里因为aisrv在心跳包里会带上是否让learner启动on-policy流程, 故需要注意的点:
        1. end_time只能增加1次, 否则进入死循环, 无法退出
        2. 理论上因为aisrv在CONFIG.on_policy_timeout_seconds时间里判断是否有on-policy流程, 故站在leaner的角度看是需要获取所有的aisrv的响应回包
           理论上该值为len(self.aisrv_zmq_client_map) * CONFIG.on_policy_timeout_seconds,
           但是配置过大导致learner的主循环阻塞, 故折中设置为2 * CONFIG.on_policy_timeout_seconds
        """
        end_time = time.time() + 2 * CONFIG.on_policy_timeout_seconds
        success_recv_aisrv_ip = []
        any_recv_aisrv_ask_on_policy_success = False
        update_end_time = False
        while time.time() < end_time:
            for aisrv_ip, zmq_client in self.aisrv_zmq_client_map.items():
                if aisrv_ip not in success_recv_aisrv_ip:
                    try:
                        recv_data = zmq_client.recv(block=False, binary=False)
                        if recv_data:
                            # 收到了aisrv让learner启动on-policy流程时, 需要发送确认响应
                            if (
                                recv_data[KaiwuDRLDefine.MESSAGE_TYPE]
                                == KaiwuDRLDefine.ON_POLICY_MESSAGE_ASK_LEARNER_TO_EXECUTE_ON_POLICY_PROCESS_REQUEST
                            ):
                                success_recv_aisrv_ip.append(aisrv_ip)

                                any_recv_aisrv_ask_on_policy_success = True
                                self.logger.info(
                                    f"train learner recv aisrv {aisrv_ip} ask to execute on-policy request success"
                                )

                            else:
                                pass

                    except Exception as e:
                        # 减少CPU争用
                        time.sleep(CONFIG.idle_sleep_second)

            """
            跳出循环的条件:
            1. 有aisrv发送on-policy请求, 并且已经满足比例的aisrv收到on-policy的请求则跳出循环
            2. 没有aisrv发送on-policy请求, 并且已经满足超时时间则跳出循环
            """
            if any_recv_aisrv_ask_on_policy_success:
                if len(success_recv_aisrv_ip) / len(self.aisrv_zmq_client_map) >= CONFIG.on_policy_quantity_ratio:
                    break

            # 如果有任何aisrv发送了on-policy的请求, 则满足最后那个aisrv发送请求的最大延长时间
            if any_recv_aisrv_ask_on_policy_success and not update_end_time:
                end_time = time.time() + CONFIG.on_policy_timeout_seconds
                update_end_time = True

        # 如果本周期内有aisrv发起了让learner去执行on-policy流程的通知, 但是个数不相等的话即接入告警, 否则走on-policy流程
        if any_recv_aisrv_ask_on_policy_success:
            if len(success_recv_aisrv_ip) / len(self.aisrv_zmq_client_map) < CONFIG.on_policy_quantity_ratio:
                keys1 = set(success_recv_aisrv_ip)
                keys2 = set(self.aisrv_zmq_client_map.keys())

                # 增加告警和容灾
                self.on_policy_learner_recv_aisrv_error_count += 1

                self.logger.error(
                    f"train process learner not recv aisrv ask on-policy request ips: {list(keys2-keys1)}"
                )
            else:
                self.on_policy_learner_recv_aisrv_success_count += 1

                # 开始执行train的操作, 然后再让learner走on-policy流程
                is_train_success = self.train()

                self.learner_on_policy_process(is_train_success)

    def on_policy_learner_connect_to_aisrv(self):
        """
        learner访问aisrv
        1. 心跳请求, 对于心跳响应的返回值不同则处理方式不同
        1.1 如果心跳响应里某个aisrv要求learner需要走on-policy流程
        1.2 如果心跳响应里没有aisrv要求learner需要走on-policy流程
        """

        if not self.aisrv_zmq_client_map:
            return

        """
        如果是需要aisrv发起来的on-policy流程, 此时需要朝aisrv发送自己的client_id
        """
        learner_send_to_aisrv_heartbeat_success_count = 0
        for aisrv_ip, zmq_client in self.aisrv_zmq_client_map.items():
            send_data = {
                KaiwuDRLDefine.MESSAGE_TYPE: KaiwuDRLDefine.ON_POLICY_MESSAGE_HEARTBEAT_REQUEST,
                KaiwuDRLDefine.MESSAGE_VALUE: KaiwuDRLDefine.ON_POLICY_MESSAGE_HEARTBEAT_REQUEST,
            }
            zmq_client.send(send_data, binary=False)
            self.logger.debug(f"train send heartbeat request to aisrv: {aisrv_ip} success")

        """
        同步等待心跳响应回包, 这里因为aisrv在心跳包里会带上是否让learner启动on-policy流程, 故需要注意的点:
        1. 计算心跳回包和计算aisrv启动on-policy的数量需要分开
        2. end_time只能增加1次, 否则进入死循环, 无法退出
        3. 理论上因为aisrv在CONFIG.on_policy_timeout_seconds时间里判断是否有on-policy流程, 故站在leaner的角度看是需要获取所有的aisrv的响应回包
           理论上该值为len(self.aisrv_zmq_client_map) * CONFIG.on_policy_timeout_seconds,
           但是配置过大导致learner的主循环阻塞, 故折中设置为2 * CONFIG.on_policy_timeout_seconds
        """
        end_time = time.time() + 2 * CONFIG.on_policy_timeout_seconds
        success_recv_aisrv_ip = []
        any_recv_aisrv_ask_on_policy_success = False
        update_end_time = False
        while time.time() < end_time:
            for aisrv_ip, zmq_client in self.aisrv_zmq_client_map.items():
                if aisrv_ip not in success_recv_aisrv_ip:
                    try:
                        recv_data = zmq_client.recv(block=False, binary=False)
                        if recv_data:
                            # 收到了aisrv让learner启动on-policy流程时, 需要发送确认响应
                            if (
                                recv_data[KaiwuDRLDefine.MESSAGE_TYPE]
                                == KaiwuDRLDefine.ON_POLICY_MESSAGE_HEARTBEAT_RESPONSE
                            ):
                                success_recv_aisrv_ip.append(aisrv_ip)

                                # 判断aisrv发送的需要learner的on-policy请求
                                if recv_data[KaiwuDRLDefine.MESSAGE_VALUE][
                                    KaiwuDRLDefine.ON_POLICY_MESSAGE_ASK_LEARNER_TO_EXECUTE_ON_POLICY_PROCESS_REQUEST
                                ]:
                                    any_recv_aisrv_ask_on_policy_success = True
                                    self.logger.info(
                                        f"train learner recv aisrv {aisrv_ip} ask to execute on-policy request success"
                                    )

                                learner_send_to_aisrv_heartbeat_success_count += 1
                                self.logger.debug(f"train learner recv aisrv {aisrv_ip}  heartbeat response success")

                            else:
                                pass

                    except Exception as e:
                        # 减少CPU争用
                        time.sleep(CONFIG.idle_sleep_second)

            """
            跳出循环的条件:
            1. 有aisrv发送on-policy请求, 并且已经满足比例的aisrv收到on-policy的请求则跳出循环
            2. 没有aisrv发送on-policy请求, 并且已经满足所有的aisrv的心跳收到请求则跳出循环
            """
            if any_recv_aisrv_ask_on_policy_success:
                if len(success_recv_aisrv_ip) / len(self.aisrv_zmq_client_map) >= CONFIG.on_policy_quantity_ratio:
                    break
            else:
                if learner_send_to_aisrv_heartbeat_success_count == len(self.aisrv_zmq_client_map):
                    break

            # 如果有任何aisrv发送了on-policy的请求, 则满足最后那个aisrv发送请求的最大延长时间
            if any_recv_aisrv_ask_on_policy_success and not update_end_time:
                end_time = time.time() + CONFIG.on_policy_timeout_seconds
                update_end_time = True

        # 如果本周期内有aisrv发起了让learner去执行on-policy流程的通知, 但是个数不相等的话即接入告警, 否则走on-policy流程
        if any_recv_aisrv_ask_on_policy_success:
            if len(success_recv_aisrv_ip) / len(self.aisrv_zmq_client_map) < CONFIG.on_policy_quantity_ratio:
                keys1 = set(success_recv_aisrv_ip)
                keys2 = set(self.aisrv_zmq_client_map.keys())

                # 增加告警和容灾
                self.on_policy_learner_recv_aisrv_error_count += 1

                self.logger.error(
                    f"train process learner not recv aisrv ask on-policy request ips: {list(keys2-keys1)}"
                )
            else:
                self.on_policy_learner_recv_aisrv_success_count += 1

                # 开始执行train的操作, 然后再让learner走on-policy流程
                is_train_success = self.train()

                self.learner_on_policy_process(is_train_success)
        else:
            if learner_send_to_aisrv_heartbeat_success_count == len(self.aisrv_zmq_client_map):
                self.logger.info(
                    "train learner recv all aisrv heartbeat response success, "
                    f"count: {learner_send_to_aisrv_heartbeat_success_count}"
                )

            else:
                # 由于无法收到aisrv的请求, 那么此时不确定aisrv的情况是怎么样, 故清空self.aisrv_zmq_client_map, 重新拉取看下效果
                self.logger.error(f"train learner not recv all aisrv heartbeat response, retry next time")

                self.aisrv_zmq_client_map.clear()
                self.on_policy_learner_get_aisrv_address()

    def learner_on_policy_process_by_aisrv_detail(self):
        """
        on-policy的流程, 从aisrv角度启动
        1. 获取所有aisrv发起的需要执行on-policy流程的个数, 即返回来的zmq信息消息数目 = 从alloc服务获取的返回的aisrv列表
        2. 如果1满足, 则开启on-ploicy流程
        3. 如果1不满足, 则等待到超时时间后即告警, 后期做容灾处理
        """

        self.on_policy_learner_recv_aisrv_on_policy_req_resp()

    def learner_on_policy_process(self, is_train_success):
        """
        on-policy需要启动流程:
        1. 由于本次训练是依靠样本消耗比, 故本次不一定能训练, 根据是否训练下面操作:
        1.1 训练成功则:
        1.1.1 清空样本池, 不会失败
        1.1.2 learner推送model文件到modelpool
        1.1.2.1 如果成功则继续剩余流程
        1.1.2.2 失败则告警指标增加
        1.1.3 learner通知aisrv最新model文件版本号
        1.1.3.1 如果成功则继续剩余流程
        1.1.3.2 如果不成功则告警, 下一步做容灾
        1.1.4 learner等待aisrv获取最新model文件版本号完毕通知
        1.14.1 如果成功则继续剩余流程
        1.1.4.2 如果不成功则告警, 下一步做容灾
        1.1.5 learner通知actor从modelpool拉取model文件
        1.1.5.1 如果成功则继续剩余流程
        1.1.5.2 如果不成功则告警, 下一步做容灾
        1.1.6 learner等待actor确认加载model文件完毕通知
        1.1.6.1 如果成功则继续剩余流程
        1.1.6.2 如果不成功则告警, 下一步做容灾
        1.2 训练不成功
        1.2.1 learner通知aisrv最新model文件版本号
        1.2.1.1 如果成功则本次完成
        1.2.1.2 如果不成功则告警, 下一步做容灾
        """

        # 清空样本池, 如果本次有进行训练才能清空样本池, 否则不需要清空, 如果强制清空, 下次learner可能会卡在reverb读写上面

        if is_train_success:
            self.replay_buffer_wrapper.reset(self.cached_local_step, self.model_wrapper.tf_sess)
            self.logger.info(f"train learner have train, so reverb reset success")
        else:
            self.logger.info(f"train learner not have train, so reverb not need reset")

        """
        消息格式:
        message_type: xxxx
        message_value: yyyy
        """
        send_data = {
            KaiwuDRLDefine.MESSAGE_TYPE: KaiwuDRLDefine.ON_POLICY_MESSAGE_MODEL_VERSION_CHANGE_REQUEST,
            KaiwuDRLDefine.MESSAGE_VALUE: self.current_sync_model_version_from_learner,
        }

        if is_train_success:
            # learner推送model文件到modelpool, 有重试机制
            learner_push_model_file_success = False
            for i in range(int(CONFIG.on_policy_error_max_retry_rounds)):
                if self.learner_push_model_to_modelpool():
                    learner_push_model_file_success = True
                    break
            """
            如果本次leaner推送到modelpool失败时, learner自身来说可以下一次再推送model文件重试, 并且可以下一次再走on-policy流程, 下面处理方法优缺点
            1. 告警指标++, 并且需要同步aisrv, actor最新的model_version
            1.1 缺点: actor上的model_version和实际的model文件不一致
            1.2 优点: aisrv上的从actor同步到的model_version和从learner同步到的model_version是一致的, aisrv的筛选样本的逻辑不会出问题
            2. 告警指标++, 同步aisrv但是不同步actor最新model_version
            2.1 优点: 减少1次learner和actor的model_version通信
            2.2 缺点: aisrv上的从actor同步到的model_version和从learner同步到的model_version是不一致的, aisrv的筛选样本的逻辑出现问题, 训练无法继续

            目前选择方法1
            """
            if not learner_push_model_file_success:
                self.logger.error(f"train process learner push_checkpoint_to_model_pool failed, so return")
                self.on_policy_push_to_modelpool_error_count += 1

            else:
                self.on_policy_push_to_modelpool_success_count += 1
                # on_policy_learner_change_model_version_cnt代表是真实的model_version次数, 故只有在真实的同步时计数
                self.on_policy_learner_change_model_version_cnt += 1

            """
            learner通知actor, aisrv更新model文件版本号, 先更新actor端, 再更新aisrv端,
            原因:
            1. 如果先更新aisrv端的model版本号, 如果actor再更新model文件失败, aisrv就开始过滤掉样本, 从而引起没有新的样本发送给learner
            2. 如果先更新actor的model文件版本号, 如果actor更新失败, 则回复给aisrv, 不进行该次的model_version更新给aisrv,
                则此轮的aisrv会按照旧的model文件版本号发送样本到learner
            """
            self.learner_send_and_recv_actor_model_version_request_and_response(send_data)

        # 无论is_train_success正确与否, 这里都需要learner和aisrv同步信息
        self.learner_send_and_recv_aisrv_model_version_request_and_response(send_data)

        self.logger.info(f"train process learner on_policy complete success")

    # learner朝aisrv发送model_version请求和收取响应
    def learner_send_and_recv_aisrv_model_version_request_and_response(self, send_data):
        if not send_data:
            return False

        for aisrv_ip, zmq_client in self.aisrv_zmq_client_map.items():
            zmq_client.send(send_data, binary=False)
            self.logger.info(
                "train process learner send model_version sync request to aisrv: "
                f"{aisrv_ip}, model_version: {self.current_sync_model_version_from_learner}"
            )

        # learner等待aisrv获取最新model文件版本号完毕通知
        learner_recv_all_aisrv_success = False
        for i in range(int(CONFIG.on_policy_error_max_retry_rounds)):
            if self.recv_model_sync_response(self.aisrv_zmq_client_map):
                learner_recv_all_aisrv_success = True
                break

        if learner_recv_all_aisrv_success:
            self.on_policy_learner_recv_aisrv_success_count += 1
            self.logger.info(f"train process learner recv all the aisrv newest model sync resp")
            return True

        else:
            self.logger.error(f"train process learner recv not all the aisrv newest model sync resp")

            # 增加告警和容灾
            self.on_policy_learner_recv_aisrv_error_count += 1
            return False

    # learner朝actor发送model_version请求和收取响应
    def learner_send_and_recv_actor_model_version_request_and_response(self, send_data):
        if not send_data:
            return False

        for actor_ip, zmq_client in self.actor_zmq_client_map.items():
            zmq_client.send(send_data, binary=False)
            self.logger.info(
                "train process learner send model_version sync request to actor: "
                f"{actor_ip}, model_version: {self.current_sync_model_version_from_learner}"
            )

        # learner等待actor确认加载model文件完成通知, 错误情况接入监控告警
        learner_recv_all_actor_success = False
        for i in range(int(CONFIG.on_policy_error_max_retry_rounds)):
            if self.recv_model_sync_response(self.actor_zmq_client_map):
                learner_recv_all_actor_success = True
                break

        if learner_recv_all_actor_success:
            self.logger.info(f"train process learner recv all the actor newest model sync resp")
            self.on_policy_learner_recv_actor_success_cnt += 1
            return True

        else:
            self.logger.error(f"train process learner recv not all the actor newest model sync resp")

            # 增加告警和容灾
            self.on_policy_learner_recv_actor_error_cnt += 1
            return False

    # learner推送model文件到modelpool去, 加上重试机制
    def learner_push_model_to_modelpool(self):
        all_push_model_success = False
        retry_count = 0

        while not all_push_model_success and retry_count < int(CONFIG.on_policy_error_retry_count_when_modelpool):
            push_model_success = self.model_file_sync_wrapper.push_checkpoint_to_model_pool(self.logger)
            if not push_model_success:
                # 如果本次失败, 则sleep下再重试, 这里重试的间隔设置大些
                time.sleep(CONFIG.idle_sleep_second * 1000)
            else:
                all_push_model_success = True
                self.logger.info(f"train learner learner_push_model_to_modelpool success")
                break

            retry_count += 1

        return all_push_model_success

    def recv_model_sync_response(self, zmq_client_map):
        """
        获取发出去的model_version同步请求的响应
        1. learner <--> aisrv
        2. learner <--> actor
        """

        if not zmq_client_map:
            return True

        # learner等待actor确认加载model文件完成通知, 错误情况接入监控告警
        success_recv_cnt = 0
        # 真正完成了model_sync_version操作的计数
        success_model_sync_cnt = 0

        retry_count = 0
        # aisrv/actor会返回结果, 但是结果里有正确和错误的区分, 故采用下面2个变量实现
        response_success_ip = {}
        model_version_change_ip = {}

        """
        重试时间即等于retry_count * CONFIG.idle_sleep_second
        1. actor是在主循环里加载, 采用默认的retry_count * CONFIG.idle_sleep_second即可
        2. aisrv的超时时间设置如下:
        2.1 如果不是按照单局或者单帧的, 采用默认的retry_count * CONFIG.idle_sleep_second即可
        2.2 如果是按照单局或者单帧的, 采用的值需要大于2 * CONFIG.on_policy_timeout_seconds
        """
        while success_recv_cnt != len(zmq_client_map) and retry_count < int(CONFIG.on_policy_error_retry_count):
            for ip, zmq_client in zmq_client_map.items():
                # 如果已经成功的不需要重复获取响应
                if ip not in response_success_ip:
                    try:
                        recv_data = zmq_client.recv(block=False, binary=False)
                        if recv_data:
                            if (
                                recv_data[KaiwuDRLDefine.MESSAGE_TYPE]
                                == KaiwuDRLDefine.ON_POLICY_MESSAGE_MODEL_VERSION_CHANGE_RESPONSE
                            ):
                                response_success_ip[ip] = ip

                                # 每个aisrv/actor明确返回model_version修改结果
                                if recv_data[KaiwuDRLDefine.MESSAGE_VALUE]:
                                    model_version_change_ip[ip] = ip
                                    success_model_sync_cnt += 1

                                success_recv_cnt += 1

                            else:
                                # 如果这里陷入重试操作, 此时aisrv发送的on-policy流程的请求可能被忽略, 故这里加上日志验证
                                self.logger.error(
                                    "train process learner model sync recv un support "
                                    f"{recv_data[KaiwuDRLDefine.MESSAGE_TYPE]}"
                                )

                    except Exception as e:
                        # 减少CPU争用
                        time.sleep(CONFIG.idle_sleep_second)

            retry_count += 1

        """
        如果收到的响应的机器数量不等发送请求的机器数量, 返回False
        如果收到的响应的机器数量和发送请求的机器数量相等, 但是不全部执行成功的, 返回False
        """
        if success_recv_cnt != len(zmq_client_map):
            keys1 = set(response_success_ip.keys())
            keys2 = set(zmq_client_map.keys())

            self.logger.error(f"train process learner model sync not recv resp ips: {keys2-keys1}")
            return False
        else:
            """
            因为此时响应包已经发送回来了, 即使内容表述的是aisrv没有执行成功, 则也返回True
            """
            if success_model_sync_cnt != len(zmq_client_map):
                keys1 = set(model_version_change_ip.keys())
                keys2 = set(zmq_client_map.keys())

                self.logger.error(f"train process learner model sync recv resp but recv error resp ips: {keys2-keys1}")
                return True

        return True

    # learner --> actor的model文件同步, 目前采用的是model pool, 后期考虑优化, 当前的actor的local step, 同步learner上的global_step
    def model_file_sync(self):
        self.logger.debug(
            f"train process after model file sync, current global step is {self.model_wrapper.get_global_step()}"
        )

    # 监控项置位
    def train_stat_reset(self):
        self.batch_train_cost_time_ms = 0
        self.sample_production_and_consumption_ratio = 0

    def set_monitor_proxy(self, monitor_proxy):
        self.monitor_proxy = monitor_proxy

    # 这里增加train的统计项
    def train_stat(self):
        """
        样本的生成速度: reverb的间隔时间里insert的样本数,注意插入次数是一直增长的, 故需要设置2个变量才能计算出差值
        样本的消耗速度: 训练的次数 * batch_size, 注意训练的次数是一直增长的, 故需要设置2个变量才能计算出差值
        样本的消耗/生产比 = 样本消耗的速度 / 样本的生产的速度
        """
        train_count = self.model_wrapper.train_stat
        reverb_insert_count = self.replay_buffer_wrapper.get_insert_stats()

        reverb_current_size = self.replay_buffer_wrapper.get_current_size()

        self.sample_product_rate = reverb_insert_count - self.last_reverb_insert_count
        self.sample_consume_rate = (train_count - self.last_train_count) * int(CONFIG.train_batch_size)

        if self.sample_product_rate == 0:
            self.sample_production_and_consumption_ratio = 0
        else:
            self.sample_production_and_consumption_ratio = self.sample_consume_rate / self.sample_product_rate

        self.last_train_count = train_count
        self.last_reverb_insert_count = reverb_insert_count

        if int(CONFIG.use_prometheus) and CONFIG.wrapper_type == KaiwuDRLDefine.WRAPPER_REMOTE:
            monitor_data = {
                KaiwuDRLDefine.MONITOR_REVERB_READY_SIZE: reverb_current_size,
                KaiwuDRLDefine.MONITOR_TRAIN_SUCCESS_CNT: self.model_wrapper.train_stat,
                KaiwuDRLDefine.MONITOR_TRAIN_GLOBAL_STEP: self.model_wrapper.get_global_step(),
                KaiwuDRLDefine.MONITOR_BATCH_TRAIN_COST_TIME_MS: self.batch_train_cost_time_ms,
                KaiwuDRLDefine.LEARNER_TCP_AISRV: actor_learner_aisrv_count(self.host, CONFIG.svr_name),
                KaiwuDRLDefine.SAMPLE_PRODUCTION_AND_CONSUMPTION_RATIO: self.sample_production_and_consumption_ratio,
                KaiwuDRLDefine.SAMPLE_PRODUCT_RATE: self.sample_product_rate,
                KaiwuDRLDefine.SAMPLE_CONSUME_RATE: self.sample_consume_rate,
            }

            if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
                monitor_data[
                    KaiwuDRLDefine.ON_POLICY_PUSH_TO_MODELPOOL_ERROR_CNT
                ] = self.on_policy_push_to_modelpool_error_count
                monitor_data[
                    KaiwuDRLDefine.ON_POLICY_PUSH_TO_MODELPOOL_SUCCESS_CNT
                ] = self.on_policy_push_to_modelpool_success_count
                monitor_data[
                    KaiwuDRLDefine.ON_POLICY_LEARNER_RECV_AISRV_ERROR_CNT
                ] = self.on_policy_learner_recv_aisrv_error_count
                monitor_data[
                    KaiwuDRLDefine.ON_POLICY_LEARNER_RECV_AISRV_SUCCESS_CNT
                ] = self.on_policy_learner_recv_aisrv_success_count
                monitor_data[
                    KaiwuDRLDefine.ON_POLICY_LEARNER_RECV_ACTOR_ERROR_CNT
                ] = self.on_policy_learner_recv_actor_error_cnt
                monitor_data[
                    KaiwuDRLDefine.ON_POLICY_LEARNER_RECV_ACTOR_SUCCESS_CNT
                ] = self.on_policy_learner_recv_actor_success_cnt

            # 按照业务数据返回的map格式直接赋值, 然后去普罗米修斯监控上设置下展示字段即可
            for key, value in self.app_monitor_data.items():
                monitor_data[key] = float(value)

            self.monitor_proxy.put_data({self.pid: monitor_data})

        # 指标复原, 计算的是周期性的上报指标
        self.train_stat_reset()

        self.logger.info(f"train process now input ready size is {self.replay_buffer_wrapper.get_current_size()}")
        self.logger.info(
            "train process now train count is "
            f"{self.model_wrapper.train_stat}, global step is "
            f"{self.model_wrapper.get_global_step()}"
        )

    # 框架运行前创建必要的文件目录
    def make_dirs(self):
        make_single_dir(CONFIG.log_dir)
        make_single_dir(CONFIG.restore_dir)
        make_single_dir(CONFIG.user_ckpt_dir)
        make_single_dir(CONFIG.summary_dir)
        make_single_dir(CONFIG.ckpt_dir)
        make_single_dir(CONFIG.pb_model_dir)
        make_single_dir(f"{CONFIG.user_ckpt_dir}/{CONFIG.app}_{CONFIG.algo}")
        make_single_dir(f"{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}")

        # 按照需要创造旁路文件
        if int(CONFIG.use_bypass):
            make_single_dir(CONFIG.bypass_dir)

    def preload_model_file(self):
        """
        预加载模式功能是指将预先训练好的baseline文件加载到KaiwuDRL里, 只是learner需要处理, actor会通过learner<-->actor之间的model文件同步在某个时间阈值后替换
        1. tensorflow, 该框架自动支持
        2. pytorch, 需要手工调用下函数

        使用方法:
        1. 需要在/data/ckpt/app_algo下放置需要设置的model文件
        2. 修改/data/ckpt/app_algo下checkpoint文件内容, 指向1中的model文件
        """

        if not int(CONFIG.preload_model):
            return

        if (
            KaiwuDRLDefine.MODEL_TENSORFLOW_SIMPLE == CONFIG.use_which_deep_learning_framework
            or KaiwuDRLDefine.MODEL_TENSORRT == CONFIG.use_which_deep_learning_framework
            or KaiwuDRLDefine.MODEL_TENSORFLOW_COMPLEX == CONFIG.use_which_deep_learning_framework
        ):
            self.logger.info(f"train tensorflow preload, not need to call function")

        elif KaiwuDRLDefine.MODEL_PYTORCH == CONFIG.use_which_deep_learning_framework:
            if not check_path_id_valid(CONFIG.preload_model_dir, CONFIG.preload_model_id):
                self.logger.error(
                    f"train pytorch preload, but preload_model_dir {CONFIG.preload_model_dir} or "
                    f"preload_model_id {CONFIG.preload_model_id} not valid, please check"
                )
                return

            self.model_wrapper.preload_model_file(CONFIG.preload_model_dir, CONFIG.preload_model_id)
            self.logger.info(
                f"train pytorch preload model file success, preload_model_dir is {CONFIG.preload_model_dir}, "
                f"preload_model_id is {CONFIG.preload_model_id}"
            )
        else:
            self.logger.error(
                f"train preload just not support {CONFIG.use_which_deep_learning_framework}, "
                f"support list is KaiwuDRLDefine.MODEL_TENSORFLOW_SIMPLE, KaiwuDRLDefine.MODEL_TENSORRT,"
                f"KaiwuDRLDefine.MODEL_TENSORFLOW_COMPLEX, KaiwuDRLDefine.MODEL_PYTORCH"
            )

    def get_replay_buffer_object(self):
        """
        主要是需要before_run下实例化了ReplayBufferWrapper后调用
        """
        return self.replay_buffer_wrapper

    def before_run(self):
        self.make_dirs()

        # 支持间隔N分钟, 动态修改配置文件
        if int(CONFIG.use_rainbow):
            from kaiwudrl.common.utils.rainbow_wrapper import RainbowWrapper

            self.rainbow_wrapper = RainbowWrapper(self.logger)

            # 第一次配置主动从七彩石拉取, 后再设置为周期性拉取
            self.rainbow_activate()
            set_schedule_event(CONFIG.rainbow_activate_per_minutes, self.rainbow_activate)

        # 根据不同启动方式来进行处理
        self.start_learner_process_by_type()

        self.process_run_count = 0

        # 获取本机IP
        self.host = get_host_ip()

        # 启动独立的进程, 负责learner与alloc交互
        if int(CONFIG.use_alloc):
            self.alloc_proxy = AllocProxy()
            self.alloc_proxy.start()

        # 需要考虑没有使用监控的场景下故提前对self.monitor_proxy做了定义
        self.monitor_proxy = None
        if int(CONFIG.use_prometheus):
            # 启动独立的进程, 负责learner与普罗米修斯交互
            self.monitor_proxy = MonitorProxy()
            self.monitor_proxy.start()

        # 注册定时器任务, 因为关键日志需要打印, 故无论需要进行普罗米修斯监控否都调用下
        set_schedule_event(CONFIG.prometheus_stat_per_minutes, self.train_stat)

        # policy_name 主要是和conf/app_conf.json设置一致
        self.policy_conf = AppConf[CONFIG.app].policies

        # 在标准化接入中, 需要引入业务自定义的workflow, 即while True循环
        if CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_STANDARD:
            self.workflow = None

        # model_wrapper, 由于ModelFileSyncWrapper和ModelFileSave需要判断是否是主learner才能进行下一步处理, 故提前到这里进行
        if CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_NORMAL:
            create_normal_model_wrapper(
                self.policy_conf,
                self.policy_model_wrapper_maps,
                self.replay_buffer_wrapper,
                self.logger,
            )

        elif CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_STANDARD:
            create_standard_model_wrapper(
                self.policy_conf,
                self.policy_model_wrapper_maps,
                self.replay_buffer_wrapper,
                self.logger,
                self.monitor_proxy,
            )

        else:
            pass

        # 因为在learner上默认只有1个agent对象, 故设置CONFIG.policy_name所在的
        self.model_wrapper = self.policy_model_wrapper_maps.get(CONFIG.policy_name)

        # model_file_sync_wrapper, actor和learner之间的Model文件同步, 采用单独的进程处理, 只有主learner进程才会执行
        if self.model_wrapper.is_chief:
            if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_OFF_POLICY:
                self.model_file_sync_wrapper = ModelFileSyncWrapper()
                self.model_file_sync_wrapper.init()

                # 该set_name下的aisrv地址个数
                self.aisrv_process_count = 0

                # 因为需要从learner获取aisrv地址
                self.alloc_util = AllocUtils(self.logger)

            elif CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
                self.current_sync_model_version_from_learner = -1
                self.model_file_sync_wrapper = ModelFileSync()
                self.model_file_sync_wrapper.make_model_dirs(self.logger)

                # 由于aisrv依赖learner先启动, 故learner启动后再去周期性的获取aisrv地址并且建立TCP连接
                set_schedule_event(
                    CONFIG.prometheus_stat_per_minutes,
                    self.on_policy_learner_get_and_connect_aisrv,
                )
                self.had_send_heartbeat_request = False

                set_schedule_event(
                    CONFIG.prometheus_stat_per_minutes,
                    self.on_policy_learner_get_and_connect_actor,
                )

                self.alloc_util = AllocUtils(self.logger)

                # 格式client_id->zmq_client对象
                self.actor_zmq_client_map = {}
                self.aisrv_zmq_client_map = {}

                # 下面是统计告警指标
                self.on_policy_push_to_modelpool_error_count = 0
                self.on_policy_push_to_modelpool_success_count = 0
                self.on_policy_learner_recv_aisrv_error_count = 0
                self.on_policy_learner_recv_aisrv_success_count = 0
                self.on_policy_learner_recv_actor_error_cnt = 0
                self.on_policy_learner_recv_actor_success_cnt = 0
                self.on_policy_learner_change_model_version_cnt = 0

            else:
                pass

        # model_file_saver, 用于保存模型文件到持久化设备, 比如COS, 采用单独的进程处理, 只有主learner进程才会执行
        if self.model_wrapper.is_chief:
            self.model_file_saver = ModelFileSave()
            self.model_file_saver.start()

        # 预先加载模型文件模式
        if int(CONFIG.preload_model):
            self.preload_model_file()

        # 启动zmq_server, 处理来自aisrv, actor的管理流, 端口设置为CONFIG.reverb_svr_port - 2, 需要考虑到zmq_server和reverb_server同时启动的情况
        self.zmq_server = ZmqServer(KaiwuDRLDefine.ALL_HOST_IP, CONFIG.reverb_svr_port - 2)
        self.zmq_server.bind()
        self.logger.info(
            f"train zmq server on learner bind at "
            f"{KaiwuDRLDefine.ALL_HOST_IP}: {CONFIG.reverb_svr_port - 2} for aisrv"
        )

        # 如果是pytorch, 则默认第一次保存文件
        if CONFIG.use_which_deep_learning_framework == KaiwuDRLDefine.MODEL_PYTORCH:
            if CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_NORMAL:
                self.model_wrapper.save_param()

            elif CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_STANDARD:
                # 清空id_list文件, 否则文件会持续增长
                clear_id_list_file(framework=True)

                # 第一次保存模型时id的默认值即0
                self.model_wrapper.save_param_by_source(source=KaiwuDRLDefine.SAVE_OR_LOAD_MODEL_By_FRAMEWORK)

                # 更新id_list文件
                update_id_list(0, framework=True)

                # 清空使用者保存的文件目录
                clear_user_ckpt_dir()

            else:
                pass

        """
        统计监控指标
        1. 批处理的训练耗时
        """
        self.batch_train_cost_time_ms = 0
        self.last_input_ready_count = 0
        self.batch_train_cost_time_ms = 0
        self.sample_production_and_consumption_ratio = 0
        self.last_reverb_insert_count = 0
        self.last_train_count = 0
        self.last_input_ready_count = 0
        self.sample_product_rate = 0
        self.sample_consume_rate = 0

        # 业务算法类监控值是个map形式
        self.app_monitor_data = {}

        # 针对来自aisrv的管理流启动单个线程处理
        t = threading.Thread(target=self.learner_process_message_by_aisrv)
        t.daemon = True
        t.start()

        # 注册SIGTERM信号处理
        if CONFIG.wrapper_type == KaiwuDRLDefine.WRAPPER_REMOTE:
            register_sigterm_handler(self.handle_sigterm, CONFIG.sigterm_pids_file)

        # 在before run最后打印启动成功日志
        self.logger.info(
            f"train process start success at {self.pid}, {self.name} trainer global step {self.local_step.value} "
            f"load app {CONFIG.app} algo {CONFIG.algo} model"
        )

        return True

    def on_policy_learner_get_and_connect_actor(self):
        """
        learner获取actor地址并且建立TCP连接, 包括下面的操作:
        1. 获取actor地址, 分是否使用alloc服务
        2. 根据1中获取actor地址情况进行处理
        2.1 如果1中获取actor地址失败, 则下次重试
        2.2 如果1中获取actor地址成功, 则本次执行
        """

        # 如果self.actor_zmq_client_map为空则走获取actor地址流程
        if not self.actor_zmq_client_map:
            self.on_policy_learner_get_actor_address()

        # 发送和接收心跳请求
        if self.actor_zmq_client_map:
            self.on_policy_learner_recv_actor_heartbeat_req_resp()

    # on-policy场景下, learner获取actor地址
    def on_policy_learner_get_actor_address(self):
        """
        1. 如果不使用alloc服务, 则直接使用本地配置, 本地配置为空则使用127.0.0.1
        2. 如果使用alloc服务, 则直接使用alloc服务
        """
        actor_address = [KaiwuDRLDefine.LOCAL_HOST_IP]
        if int(CONFIG.use_alloc):
            self.alloc_util.registry()
            actor_address = self.alloc_util.get_all_address_by_srv_name(KaiwuDRLDefine.SERVER_ACTOR)
            if not actor_address:
                self.logger.error(f"train get actor_address error, retry next time")
                return
            else:
                self.logger.info(f"train get actor_address success,  actor address: {actor_address}")
        else:
            self.logger.info(f"train set use_alloc False, so actor use {KaiwuDRLDefine.LOCAL_HOST_IP}")

        for address in actor_address:
            client_id = get_uuid()
            actor_ip = address.split(":")[0]
            actor_port = int(CONFIG.zmq_server_port) + 100
            zmq_client = ZmqClient(str(client_id), actor_ip, actor_port)
            zmq_client.connect()
            self.actor_zmq_client_map[f"{actor_ip}:{actor_port}"] = zmq_client

    # on-policy场景下, learner与actor地址建立连接, 周期性的发送/接收心跳保活请求/响应
    def on_policy_learner_recv_actor_heartbeat_req_resp(self):
        if not self.actor_zmq_client_map:
            return

        # 因为心跳请求的send_data是一致的, 故可以放在循环外面
        send_data = {
            KaiwuDRLDefine.MESSAGE_TYPE: KaiwuDRLDefine.ON_POLICY_MESSAGE_HEARTBEAT_REQUEST,
            KaiwuDRLDefine.MESSAGE_VALUE: KaiwuDRLDefine.ON_POLICY_MESSAGE_HEARTBEAT_REQUEST,
        }

        learner_send_to_actor_heartbeat_success_count = 0
        for actor_ip, zmq_client in self.actor_zmq_client_map.items():
            zmq_client.send(send_data, binary=False)
            self.logger.debug(f"train send heartbeat request to actor: {actor_ip} success")

            # 同步等待心跳响应回包
            retry_count = 0
            while retry_count < int(CONFIG.on_policy_error_retry_count):
                try:
                    recv_data = zmq_client.recv(block=False, binary=False)
                    if recv_data:
                        if (
                            recv_data[KaiwuDRLDefine.MESSAGE_TYPE]
                            == KaiwuDRLDefine.ON_POLICY_MESSAGE_HEARTBEAT_RESPONSE
                        ):
                            self.logger.debug(f"train recv heartbeat response to actor: {actor_ip} success")
                            learner_send_to_actor_heartbeat_success_count += 1
                            break
                except Exception as e:
                    # 减少CPU争用
                    time.sleep(CONFIG.idle_sleep_second)

                retry_count += 1

        # 以为心跳的请求频率比较高, 打印日志比较耗时, 故采用debug日志
        if learner_send_to_actor_heartbeat_success_count == len(self.actor_zmq_client_map):
            self.logger.debug(
                f"train learner recv all actor heartbeat response success, "
                f"count: {learner_send_to_actor_heartbeat_success_count}"
            )

        else:
            # 由于无法收到actor的请求, 那么此时不确定actor的情况是怎么样, 故清空self.actor_zmq_client_map, 重新拉取看下效果
            self.logger.error(
                f"train learner not recv all actor heartbeat response, retry next time, "
                f"learner_send_to_actor_heartbeat_success_count "
                f"{learner_send_to_actor_heartbeat_success_count} != "
                f"len(actor_zmq_client_map) {len(self.actor_zmq_client_map)}"
            )

            self.actor_zmq_client_map.clear()
            self.on_policy_learner_get_actor_address()

    def on_policy_learner_get_and_connect_aisrv(self):
        """
        learner获取aisrv地址并且建立TCP连接, 包括下面的操作:
        1. 获取aisrv地址, 分是否使用alloc服务
        2. 根据1中获取aisrv地址情况进行处理
        2.1 如果1中获取aisrv地址失败, 则下次重试
        2.2 如果1中获取aisrv地址成功, 则本次执行
        """

        # 如果self.aisrv_zmq_client_map为空则走获取aisrv地址流程
        if not self.aisrv_zmq_client_map:
            self.on_policy_learner_get_aisrv_address()

        # 发送心跳请求和响应
        if self.aisrv_zmq_client_map and not self.had_send_heartbeat_request:
            self.on_policy_learner_recv_aisrv_heartbeat_req_resp()

            self.had_send_heartbeat_request = True

    def learner_get_aisrv_address(self):
        """
        learner获取aisrv地址, 只是获取到地址, 不会进行建立连接
        """

        """
        1. 如果不使用alloc服务, 则直接使用本地配置, 本地配置为空则使用127.0.0.1
        2. 如果使用alloc服务, 则直接使用alloc服务
        """

        if int(CONFIG.use_alloc):
            self.alloc_util.registry()
            # on-policy情况下learner需要启动与aisrv的通信, 采用在aisrv 8000端口号 + 100的端口上监听, learner为client, aisrv为server
            aisrv_address = self.alloc_util.get_all_address_by_srv_name(KaiwuDRLDefine.SERVER_AISRV)
            if not aisrv_address:
                self.logger.error(f"train get aisrv_address error, retry next time")
                self.aisrv_process_count = 0
                return None
            else:
                self.logger.info(f"train get alloc aisrv_address success, aisrv address: {aisrv_address}")
        else:
            aisrv_default_address = CONFIG.aisrv_default_address
            if aisrv_default_address:
                original_aisrv_address = aisrv_default_address.split(",")
            else:
                original_aisrv_address = [f"{KaiwuDRLDefine.LOCAL_HOST_IP}:{CONFIG.aisrv_server_port}"]

            # 注意需要检测配置项
            aisrv_address = [item for item in original_aisrv_address if item]

            self.logger.info(f"train get default aisrv_address success, aisrv address: {aisrv_address}")

        # aisrv是分多个对局的, 每个aisrv * 对局数目
        self.aisrv_process_count = len(aisrv_address) * CONFIG.aisrv_connect_to_kaiwu_env_count

        return aisrv_address

    # on-policy场景下, learner从alloc服务获取aisrv的地址
    def on_policy_learner_get_aisrv_address(self):

        aisrv_address = self.learner_get_aisrv_address()
        if aisrv_address is None:
            self.logger.error(f"train on_policy_learner_get_aisrv_address, aisrv_address is None")
            return

        for address in aisrv_address:
            client_id = get_uuid()
            aisrv_ip = address.split(":")[0]
            aisrv_port = int(CONFIG.aisrv_server_port) + 100
            zmq_client = ZmqClient(str(client_id), aisrv_ip, aisrv_port, duplex=True)
            zmq_client.connect()

            self.aisrv_zmq_client_map[f"{aisrv_ip}:{aisrv_port}"] = zmq_client

    def save_model_detail(self, ip, path, id):
        """
        处理来自aisrv的save_model请求详细执行过程
        """
        path = f"{CONFIG.user_ckpt_dir}/{CONFIG.app}_{CONFIG.algo}/"

        # 注意此时是业务调用的
        try:
            self.model_wrapper.save_param_by_source(
                path=path,
                id=id,
                source=KaiwuDRLDefine.SAVE_OR_LOAD_MODEL_By_USER,
            )
            self.logger.info(f"train learner save_param_by_framework is success, ip is {ip}")
            return True

        except Exception as e:
            self.logger.error(f"train learner save_param_by_framework is failed, ip is {ip}")

            return False

    def learner_process_message_by_aisrv(self):
        """
        收集来自aisrv的zmq请求, 分为下面情况:
        1. 如果是save_model, 则执行save_model操作

        下面是规则:
        1. 针对不同的aisrv来让同一个learner执行save_model的操作
        1.1 在时间间隔内如果获取到第一个即执行
        1.2 在最大时间间隔内如果1.1的aisrv持续的进行save_model时则直接执行, 否则转1.3
        1.3 重新接收第一个需要执行的aisrv, 然后更新时间
        """

        # 用于处理来自aisrv的save_model请求的限制
        last_save_model_aisrv_ip = None
        last_save_model_time = 0

        # 当前接收到aisrv需要退出的进程数量
        current_aisrv_process_stop_count = 0

        # 标志是否有aisrv发送过process_stop的请求, 然后周期性判断超时退出
        had_recv_aisrv_stop_request = False
        last_recv_aisrv_stop_request_time = 0
        had_learner_process_stop = False

        while True:
            # 下面是learner超时退出逻辑
            if had_recv_aisrv_stop_request and had_learner_process_stop:
                now = time.time()
                if now - last_recv_aisrv_stop_request_time > CONFIG.aisrv_process_stop_timeout_seconds:

                    # 达到超时条件
                    error_code = KaiwuDRLDefine.DOCKER_EXIT_CODE_TIMEOUT

                    # 进程退出
                    self.logger.info(
                        f"train learner recv aisrv "
                        f"now {now} - "
                        f"last_recv_aisrv_stop_request_time {last_recv_aisrv_stop_request_time} "
                        f">= {CONFIG.aisrv_process_stop_timeout_seconds}, so exit"
                    )
                    self.learner_process_stop(error_code)

            try:
                # 收到来自aisrv的请求
                client_id, message = self.zmq_server.recv(block=False, binary=False)
                if message:
                    message_type = message.get(KaiwuDRLDefine.MESSAGE_TYPE)
                    message_value = message.get(KaiwuDRLDefine.MESSAGE_VALUE)
                    if message_type == KaiwuDRLDefine.MESSAGE_SAVE_MODEL:
                        ip = message_value.get("ip")
                        path = message_value.get("path")
                        id = message_value.get("id")
                        self.logger.info(f"train learner recv save_model from aisrv {ip}")

                        send_data = {
                            KaiwuDRLDefine.MESSAGE_TYPE: KaiwuDRLDefine.MESSAGE_SAVE_MODEL,
                            KaiwuDRLDefine.MESSAGE_VALUE: True,
                        }
                        # 需要先执行zmq回包, 再进行处理
                        self.zmq_server.send(str(client_id), send_data, binary=False)
                        self.logger.info(f"train learner send save_model result to aisrv {ip}")

                        now = time.time()

                        # 如果没有上一次的保存模型请求，或者上一次的请求来自同一个ip，执行保存模型
                        if not last_save_model_aisrv_ip or last_save_model_aisrv_ip == ip:
                            # 然后执行保存用户model文件的操作
                            if self.save_model_detail(ip, path, id):
                                last_save_model_aisrv_ip = ip
                                last_save_model_time = now

                        # 如果上一次的请求来自不同的ip，但已经超过了最大等待时间，也执行保存模型
                        else:
                            if (
                                now - last_save_model_time
                                >= CONFIG.choose_aisrv_to_load_model_or_save_model_max_time_seconds
                            ):
                                # 然后执行保存用户model文件的操作
                                if self.save_model_detail(ip, path, id):
                                    last_save_model_aisrv_ip = ip
                                    last_save_model_time = now

                    elif message_type == KaiwuDRLDefine.MESSAGE_PROCESS_STOP:
                        ip = message_value.get("ip")
                        error_code = message_value.get("error_code")
                        self.logger.info(f"train learner recv process_stop from aisrv {ip}")
                        send_data = {
                            KaiwuDRLDefine.MESSAGE_TYPE: KaiwuDRLDefine.MESSAGE_PROCESS_STOP,
                            KaiwuDRLDefine.MESSAGE_VALUE: True,
                        }

                        # 需要先执行zmq回包, 再进行处理
                        self.zmq_server.send(str(client_id), send_data, binary=False)
                        self.logger.info(f"train learner send process_stop result to aisrv {ip}")

                        had_recv_aisrv_stop_request = True
                        last_recv_aisrv_stop_request_time = time.time()

                        # 因为此时能发出来process_stop请求的aisrv是已经启动了的, 故这里重新拉取下地址即得到此时活着的aisrv进程
                        self.learner_get_aisrv_address()

                        """
                        error_code场景:
                        1. 如果是错误退出, 则只要有1个aisrv上报该错误码即需要learner退出
                        2. 如果是正常退出, 则需要按照比例来退出aisrv_process_stop_quantity_ratio, 默认是100%
                        """
                        if error_code > 0:
                            # 进程退出
                            self.logger.info(f"train learner recv error_code {error_code} from {ip}, so exit")
                            self.learner_process_stop(error_code)

                            had_learner_process_stop = True
                        else:
                            current_aisrv_process_stop_count += 1

                            if self.aisrv_process_count == 0:
                                self.logger.info(f"train learner self.aisrv_process_count is 0, so exit")
                                self.learner_process_stop(error_code)

                                had_learner_process_stop = True

                            # 达到比例退出
                            else:
                                if (
                                    current_aisrv_process_stop_count / self.aisrv_process_count
                                    >= CONFIG.aisrv_process_stop_quantity_ratio
                                ):
                                    # 进程退出
                                    self.logger.info(
                                        f"train learner recv aisrv "
                                        f"{current_aisrv_process_stop_count} / {self.aisrv_process_count} "
                                        f">= {CONFIG.aisrv_process_stop_quantity_ratio}, so exit"
                                    )
                                    self.learner_process_stop(error_code)

                                    had_learner_process_stop = True
                    else:
                        self.logger.error(f"train learner recv unknown message_type {message_type}")

            except Exception as e:
                # sleep下减少CPU损耗
                time.sleep(CONFIG.idle_sleep_second)

    def learner_process_stop(self, error_code):
        """
        learner进程的退出, 包括自己的python3进程, modelpool进程, 但是如果此时就直接退出其他进程如modelpool进程可能导致有其他进程上报model文件失败
        故统一调整到所有的进程是在被动退出时退出
        """

        # 写process_stop文件, 里面写error_code
        process_stop_write_file(error_code)

    def train(self):
        """
        训练的规则:
        1. 当reverb设置最大的size, 采用FIFO模式
        2. 当满足batch_size即开始训练, 对reverb不做主动清空操作, 从reverb里拿取的数据是随机的, 这样增加了训练次数, 新的数据进来采用FIFO去替换掉旧的
        """

        reverb_insert_count = self.replay_buffer_wrapper.get_insert_stats()
        if CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_REVERB:
            current_size = self.replay_buffer_wrapper._replay_buffer.total_size(
                self.replay_buffer_wrapper._reverb_client
            )
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_ZMQ:
            current_size = self.replay_buffer_wrapper._replay_buffer.total_size()
        else:
            current_size = 0

        # 标志本次是否真实的train
        is_train_success = False

        """
        这里需要区分下:
        1. 如果是on-policy, 必须等current_size大于batch_size才能进入到self.train_detail逻辑,
            否则因为aisrv在等learner的on-policy响应, learner在等aisrv产生样本, 就出现死锁
        2. 如何是off-policy, 不需要current_size大于batch_size, 当learner在读取reverb样本时, 阻塞了aisrv也能给learner发送样本
        """
        # 步骤2, 从reverb server获取样本数据
        """
        learner满足训练条件的情况:
        1. off-policy
        1.1 积累一定数量
        1.2 大于等于样本消耗比(如果每次加上这个判断会导致在低峰时无法训练, 而在高峰时才能训练, 减少了训练次数, 故在某些场景里得考虑去掉该判断)
        1.3 大于batch_size
        2. on-policy
        2.1 大于batch_size
        """
        # if self.input_ready():
        condition = False
        if CONFIG.learner_train_by_while_true:
            # 如果采用while True循环方式则使用sleep方式调整样本消耗比
            time.sleep(CONFIG.learner_train_sleep_seconds)
            condition = True

        elif CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_OFF_POLICY:
            condition = (current_size >= int(CONFIG.replay_buffer_capacity) // int(CONFIG.preload_ratio)) and (
                reverb_insert_count - self.last_input_ready_count
            ) >= (int(CONFIG.train_batch_size) / int(CONFIG.production_consume_ratio))

        elif CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
            condition = current_size >= int(CONFIG.train_batch_size)

        else:
            pass

        if condition:
            # 步骤3, 训练
            self.train_detail()

            if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_OFF_POLICY:
                self.last_input_ready_count = reverb_insert_count
            is_train_success = True

        return is_train_success

    def run_once(self):
        """
        learner的单次流程如下:
        1. 执行定时器操作
        2. 执行训练步骤
        3. on-policy情况下, 执行从aisrv开始的流程
        """

        # 步骤1, 启动定时器操作, 定时器里执行记录统计信息
        schedule.run_pending()

        """
        步骤2, 执行训练, 主要是下面的情况:
        1. 如果是on-policy
        1.1 如果是按照learner角度多帧的, 直接训练
        1.2 如果是按照aisrv角度单帧/单局的, 不要调用self.train训练, 而是采用self.learner_on_policy_process_by_aisrv推动
        1.3 其他的情况, 后期扩展, 直接训练
        2. 如果是off-policy, 直接训练
        3. 其他的情况, 后期扩展, 直接训练
        """
        if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
            if CONFIG.on_policy_by_way == KaiwuDRLDefine.ALGORITHM_ON_POLICY_WAY_TIME_INTERVAL:
                self.train()
            elif (
                CONFIG.on_policy_by_way == KaiwuDRLDefine.ALGORITHM_ON_POLICY_WAY_EPISODE
                or CONFIG.on_policy_by_way == KaiwuDRLDefine.ALGORITHM_ON_POLICY_WAY_STEP
            ):
                pass
            else:
                self.train()

        elif CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_OFF_POLICY:
            self.train()

        else:
            self.train()

        # 步骤3, 收集从aisrv来的发起on-policy请求
        self.learner_on_policy_process_by_aisrv()

        # Model文件保存, 同步已经采用单个进程方式进行

    def input_tensors(self):
        return self.replay_buffer_wrapper.input_tensors()

    def input_ready(self):
        """
        learner满足训练条件的情况:
        1. off-policy
        1.1 积累一定数量
        1.2 大于等于样本消耗比
        1.3 大于batch_size
        2. on-policy
        2.1 大于batch_size
        """

        return self.replay_buffer_wrapper.input_ready(None)

    def loop(self):
        if not self.before_run():
            self.logger.info("train before_run failed, so return")
            return

        while not self.model_wrapper.should_stop():
            try:
                self.run_once()

                # 短暂sleep, 规避容器里进程CPU使用率100%问题
                self.process_run_count += 1
                if self.process_run_count % CONFIG.idle_sleep_count == 0:
                    time.sleep(CONFIG.idle_sleep_second)

                    # process_run_count置0, 规避溢出
                    self.process_run_count = 0

            except Exception as e:
                self.logger.error(
                    f"train process failed to run {self.name} trainer. exit. "
                    f"Error is: {e}, traceback.print_exc() is {traceback.format_exc()}"
                )
                break

        self.model_wrapper.close()
        self.logger.info("train self.server.stop success")

        # 非on-policy的才需要主动关闭self.model_file_sync_wrapper
        if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_OFF_POLICY:
            self.model_file_saver.stop()
            self.logger.info("train self.model_wrapper.close success")

            self.model_file_sync_wrapper.stop()
            self.logger.info("train self.model_file_sync_wrapper.stop success")

    def handle_sigterm(self, sig, frame):
        # 已经创建model_wrapper,并且为主learner进程
        if hasattr(self, "model_wrapper") and self.model_wrapper.is_chief:
            self.logger.info(f"on_policy_trainer {os.getpid()} is starting to handle the SIGTERM signal.")
            self.model_wrapper.save_param_by_source(source=KaiwuDRLDefine.SAVE_OR_LOAD_MODEL_BY_SIGTERM)
            # 处理完保存最新模型,等待其他进程工作,避免pod提前退出
            time.sleep(CONFIG.handle_sigterm_sleep_seconds)
        else:
            self.logger.info(f"on_policy_trainer {os.getpid()} is not chief.")
            time.sleep(CONFIG.handle_sigterm_sleep_seconds)
