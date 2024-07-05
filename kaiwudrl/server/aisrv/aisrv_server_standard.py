#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file aisrv_server_standard.py
# @brief
# @author kaiwu
# @date 2023-11-28


import json
import multiprocessing
import datetime
import os
import traceback
import schedule
import time
import copy
import flatbuffers
import collections
from kaiwudrl.common.monitor.monitor_proxy_process import MonitorProxy
from kaiwudrl.common.logging.kaiwu_logger import KaiwuLogger, g_not_server_label
from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.common.alloc.alloc_proxy import AllocProxy, AllocUtils
from kaiwudrl.common.utils.common_func import (
    is_list_eq,
    list_diff,
    set_schedule_event,
    actor_learner_aisrv_count,
    get_host_ip,
    Context,
    make_single_dir,
    python_exec_shell,
    register_sigterm_handler,
)
from kaiwudrl.common.ipc.zmq_util import ZmqServer
from kaiwudrl.common.utils.slots import Slots
from kaiwudrl.common.config.app_conf import AppConf
from kaiwudrl.server.aisrv.flatbuffer.kaiwu_msg_helper import KaiwuMsgHelper
from kaiwudrl.server.aisrv.msg_buff import MsgBuff
from kaiwudrl.server.aisrv.flatbuffer.kaiwu_msg import ReqMsg
from kaiwudrl.common.checkpoint.model_file_sync_wrapper import ModelFileSyncWrapper


class AiServer(multiprocessing.Process):
    def __init__(
        self,
    ) -> None:
        super(AiServer, self).__init__()

        # 设置日志Log配置
        self.logger = KaiwuLogger()
        self.current_pid = os.getpid()
        self.logger.setLoggerFormat(
            f"/{CONFIG.svr_name}/aiserver_pid{self.current_pid}_log_"
            f"{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log",
            "AiServer",
        )

        # 获取本机IP
        self.host = get_host_ip()

    # 业务的上报, aisrv主线程
    def aisrv_main_process_stat(self):
        if int(CONFIG.use_prometheus):
            monitor_data = {}

            # 无论什么场景下都需要上报的监控值
            monitor_data[KaiwuDRLDefine.AISRV_TCP_BATTLESRV] = actor_learner_aisrv_count(self.host, CONFIG.svr_name)

            if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
                monitor_data = {}
                monitor_data[
                    KaiwuDRLDefine.MONITOR_AISRV_ON_POLICY_KAIWU_RL_HELPER_PAUSE_ERROR_COUNT
                ] = self.all_kaiwu_rl_helper_pause_error_count
                monitor_data[
                    KaiwuDRLDefine.MONITOR_AISRV_ON_POLICY_KAIWU_RL_HELPER_PAUSE_SUCCESS_COUNT
                ] = self.all_kaiwu_rl_helper_pause_success_count
                monitor_data[
                    KaiwuDRLDefine.MONITOR_AISRV_ON_POLICY_KAIWU_RL_HELPER_CONTINUE_ERROR_COUNT
                ] = self.all_kaiwu_rl_helper_continue_error_count
                monitor_data[
                    KaiwuDRLDefine.MONITOR_AISRV_ON_POLICY_KAIWU_RL_HELPER_CONTINUE_SUCCESS_COUNT
                ] = self.all_kaiwu_rl_helper_continue_success_count
                monitor_data[
                    KaiwuDRLDefine.MONITOR_AISRV_ON_POLICY_AISRV_CHANGE_MODEL_VERSION_ERROR_COUNT
                ] = self.aisrv_change_model_version_error_count
                monitor_data[
                    KaiwuDRLDefine.MONITOR_AISRV_ON_POLICY_AISRV_CHANGE_MODEL_VERSION_SUCCESS_COUNT
                ] = self.aisrv_change_model_version_success_count

            self.monitor_proxy.put_data({self.current_pid: monitor_data})

    def check_param(self):
        """
        进程启动前配置参数检测
        1. 规则1, 如果是设置了self-play模式, 但是app文件里设置的policy是1个, 则报错
        2. 规则2, 如果是设置了非self-play模式, 但是app文件里设置的policy是2个, 则报错
        3. 规则3, 如果是设置了self-play模式, 但是aisrv.toml文件里设置的actor_addrs/learner_addrs的policy是1个, 则报错
        4. 规则4, 如果是设置了非self-play模式, 但是aisrv.toml文件里设置的actor_addrs/learner_addrs的policy是2个, 则报错
        5. 规则5, 如果是设置了on-policy模式, 但是同时是小规模模式, 则报错
        """

        actor_addrs = CONFIG.actor_addrs
        learner_addrs = CONFIG.learner_addrs

        if int(CONFIG.self_play):
            if len(AppConf[CONFIG.app].policies) == 1:
                self.logger.error(f"AiServer self-play模式, 但是配置的policy维度为1, 请修改配置后重启进程")
                return False

            if len(actor_addrs) == 1 or len(learner_addrs) == 1:
                self.logger.error(
                    f"AiServer self-play模式, 但是配置的aisrv.toml的actor_addrs/learner_addrs的policy维度为1, 请修改配置后重启进程"
                )
                return False

        else:
            if len(AppConf[CONFIG.app].policies) == 2:
                self.logger.error(f"AiServer 非self-play模式, 但是配置的policy维度为2, 请修改配置后重启进程")
                return False

            if len(actor_addrs) == 2 or len(learner_addrs) == 2:
                self.logger.error(
                    f"AiServer 非self-play模式, 但是配置的aisrv.toml的actor_addrs/learner_addrs的policy维度为2, 请修改配置后重启进程"
                )
                return False

        if (
            CONFIG.predict_local_or_remote == KaiwuDRLDefine.PREDICT_AS_LOCAL
            and CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY
        ):
            self.logger.error(f"AiServer 配置了小集群模式但是又设置了on-policy模式, 暂不支持, 请修改配置后重启进程")
            return False

        return True

    # aisrv在处理actor和learner的动态扩缩容逻辑
    def aisrv_with_new_actor_learner_change(self):
        if not CONFIG.actor_learner_expansion:
            return

    # aisrv从alloc服务获取kaiwu_env的IP地址
    def get_kaiwu_env_ip_from_alloc(self):
        # 重试CONFIG.socket_retry_times次, 每次sleep CONFIG.alloc_process_per_seconds获取actor和learner地址
        retry_num = 0
        kaiwu_env_address = []
        while retry_num < CONFIG.socket_retry_times:
            kaiwu_env_address = self.alloc_util.get_kaiwu_env_ip(CONFIG.set_name, CONFIG.self_play_set_name)
            if not kaiwu_env_address:
                time.sleep(int(CONFIG.socket_timeout))
                retry_num += 1
            else:
                break

        # 如果超过重试次数, 则放弃从alloc获取地址, 从本地配置文件启动
        if retry_num >= CONFIG.socket_retry_times:
            self.logger.error(
                f"AiServer server get kaiwu_env address retry times more than "
                f"{CONFIG.socket_retry_times}, will start with configure file"
            )
            return None

        return kaiwu_env_address

    # aisrv朝alloc服务的注册函数, 需要先注册才能拉取地址
    def aisrv_registry_to_alloc(self):
        # 需要先注册本地aisrv地址后, 再拉取actor, learner地址
        code, msg = self.alloc_util.registry()
        if code:
            self.logger.info(f"AiServer alloc interact registry success")
            return True
        else:
            self.logger.error(f"AiServer alloc interact registry fail, will retry next time, error_code is {msg}")
            return False

    def get_actor_learner_ip_from_alloc(self):
        """
        增加aisrv从alloc获取IP地址的逻辑, 为了和以前从配置文件加载的方式结合, 采用操作步骤如下:
        1. 每隔CONFIG.alloc_process_per_seconds拉取, 最大CONFIG.socket_retry_times次后报错, 当返回有具体的数据则跳出循环
        2. 针对返回的actor和learner地址, 修改内存和配置文件里的值
        """

        if CONFIG.predict_local_or_remote == KaiwuDRLDefine.PREDICT_AS_REMOTE:
            # 重试CONFIG.socket_retry_times次, 每次sleep CONFIG.alloc_process_per_seconds获取actor和learner地址
            retry_num = 0
            while retry_num < CONFIG.socket_retry_times:
                if not int(CONFIG.self_play):
                    (
                        actor_address,
                        learner_address,
                        _,
                        _,
                    ) = self.alloc_util.get_actor_learner_ip(CONFIG.set_name, None)
                    if not actor_address or not learner_address:
                        time.sleep(int(CONFIG.socket_timeout))
                        retry_num += 1
                    else:
                        break
                else:
                    # 对于self-play模式, self_play_set下的learner不是强要求的
                    (
                        self_play_actor_address,
                        self_play_old_actor_address,
                        self_play_learner_address,
                        self_play_old_learner_address,
                    ) = self.alloc_util.get_actor_learner_ip(CONFIG.set_name, CONFIG.self_play_set_name)

                    if not self_play_actor_address or not self_play_learner_address or not self_play_old_actor_address:
                        time.sleep(int(CONFIG.socket_timeout))
                        retry_num += 1
                    else:
                        break

            # 如果超过重试次数, 则放弃从alloc获取地址, 从本地配置文件启动
            if retry_num >= CONFIG.socket_retry_times:
                self.logger.error(
                    f"AiServer server get actor and learner address retry times more than "
                    f"{CONFIG.socket_retry_times}, will start with configure file"
                )
                return

            # 修改配置文件
            if not int(CONFIG.self_play):
                self.change_configure_content(actor_address, learner_address, None, None, None, None)
            else:
                self.change_configure_content(
                    None,
                    None,
                    self_play_actor_address,
                    self_play_learner_address,
                    self_play_old_actor_address,
                    self_play_old_learner_address,
                )
        else:
            # 重试CONFIG.socket_retry_times次, 每次sleep CONFIG.alloc_process_per_seconds获取learner地址
            retry_num = 0
            while retry_num < CONFIG.socket_retry_times:
                if not int(CONFIG.self_play):
                    learner_address, _ = self.alloc_util.get_learner_ip(CONFIG.set_name, None)
                    if not learner_address:
                        time.sleep(int(CONFIG.socket_timeout))
                        retry_num += 1
                    else:
                        break
                else:
                    # 对于self-play模式, self_play_set下的learner不是强要求的
                    (
                        self_play_learner_address,
                        self_play_old_learner_address,
                    ) = self.alloc_util.get_learner_ip(CONFIG.set_name, CONFIG.self_play_set_name)
                    if not self_play_learner_address:
                        time.sleep(int(CONFIG.socket_timeout))
                        retry_num += 1
                    else:
                        break

            # 如果超过重试次数, 则放弃从alloc获取地址, 从本地配置文件启动
            if retry_num >= CONFIG.socket_retry_times:
                self.logger.error(
                    f"AiServer server get actor and learner address retry times more than "
                    f"{CONFIG.socket_retry_times}, will start with configure file"
                )
                return

            # 修改配置文件
            if not int(CONFIG.self_play):
                # 此处需要针对设置值
                actor_address = [f"{KaiwuDRLDefine.LOCAL_HOST_IP}:{CONFIG.zmq_server_port}"]
                self.change_configure_content(actor_address, learner_address, None, None, None, None)
            else:
                self_play_actor_address = [f"{KaiwuDRLDefine.LOCAL_HOST_IP}:{CONFIG.zmq_server_port}"]
                self_play_old_actor_address = [f"{KaiwuDRLDefine.LOCAL_HOST_IP}:{CONFIG.zmq_server_port}"]
                self.change_configure_content(
                    None,
                    None,
                    self_play_actor_address,
                    self_play_learner_address,
                    self_play_old_actor_address,
                    self_play_old_learner_address,
                )

    # C++ 常驻进程进程配置文件修改
    def save_to_file(self, process_name, to_change_key_values):
        if not to_change_key_values or not process_name:
            return

        # 先删除actor_addrs,learner_addrs,self_play, actor_proxy_num, learner_proxy_num
        cmd = (
            f"sed -i '/actor_addrs/d' {CONFIG.cpp_aisrv_configure}; "
            f"sed -i '/learner_addrs/d' {CONFIG.cpp_aisrv_configure}; "
            f"sed -i '/self_play/d' {CONFIG.cpp_aisrv_configure}; "
            f"sed -i '/actor_proxy_num/d' {CONFIG.cpp_aisrv_configure}; "
            f"sed -i '/learner_proxy_num/d' {CONFIG.cpp_aisrv_configure};"
        )
        result_code, result_str = python_exec_shell(cmd)
        if result_code:
            self.logger.error(f"AiServer python_exec_shell failed, cmd is {cmd}, error msg is {result_str}")
            return

        # 由于self_play是在main里配置, 这里根据返回的actor_addrs和learner_addrs来决定其值
        actor_addrs_json = json.loads(to_change_key_values.get("actor_addrs"), strict=False)
        self_play = 0
        if len(actor_addrs_json) == 2:
            self_play = 1
        to_change_key_values["self_play"] = self_play

        # 去掉actor_proxy_num和learner_proxy_num参数
        del to_change_key_values["actor_proxy_num"]
        del to_change_key_values["learner_proxy_num"]

        # 追加文件写操作
        with open(CONFIG.cpp_aisrv_configure, "a", encoding=KaiwuDRLDefine.UTF_8) as f:
            for key, value in to_change_key_values.items():
                # gflags严格要求key=value形式, 不能留空格
                f.write(f"--{key}={value}\n")
                self.logger.info(f"AiServer {CONFIG.cpp_aisrv_configure} {key} {value}")

        self.logger.info(f"AiServer {CONFIG.cpp_aisrv_configure} CONFIG save_to_file success")

    def change_configure_content(
        self,
        actor_addrs,
        learner_addrs,
        self_play_actor_address,
        self_play_learner_address,
        self_play_old_actor_address,
        self_play_old_learner_address,
    ):
        """
        修改conf/system/aisrv_system.toml里的配置项目, 如下:
        1. actor_addrs
        2. actor_proxy_num
        3. learner_addrs
        4. learner_proxy_num
        5. self_play_actor_proxy_num
        6. self_play_old_actor_proxy_num
        7. self_play_learner_proxy_num
        8. self_play_old_learner_proxy_num
        """

        # 写回配置文件内容
        to_change_key_values = {}

        # 将当前的配置文件的内容读成json串, 内存修改后, 再写回json内容, 如果解析json串出错, 则提前报错返回
        try:
            old_actor_address_map = copy.deepcopy(CONFIG.actor_addrs)
            old_learner_address_map = copy.deepcopy(CONFIG.learner_addrs)

            # 如果是非self-play, 需要删除掉CONFIG.self_play_old_policy对应的数据
            if not int(CONFIG.self_play):
                if CONFIG.self_play_old_policy in old_actor_address_map:
                    del old_actor_address_map[CONFIG.self_play_old_policy]
                if CONFIG.self_play_old_policy in old_learner_address_map:
                    del old_learner_address_map[CONFIG.self_play_old_policy]

        except Exception as e:
            self.logger.error(
                f"AiServer get actor and learner address from conf failed, error is {str(e)}",
                g_not_server_label,
            )

            return

        """
        处理实例如下:
        actor_addrs = {"train_one": ["127.0.0.1:8001"], "train_two": ["127.0.0.1:8002"]}
        learner_addrs = {"train_one": ["127.0.0.1:9000"], "train_two": ["127.0.0.1:9001"]}
        """

        if not int(CONFIG.self_play):
            if not actor_addrs and not learner_addrs:
                return

            # 如果actor_addrs不空则处理, 否则跳过
            if actor_addrs:
                actor_proxy_num = len(actor_addrs)
                old_actor_address_map[CONFIG.policy_name] = actor_addrs
                to_change_key_values["actor_proxy_num"] = actor_proxy_num
                to_change_key_values["actor_addrs"] = old_actor_address_map

            # 如果learner_addrs不空则处理, 否则跳过
            if learner_addrs:
                learner_proxy_num = len(learner_addrs)
                old_learner_address_map[CONFIG.policy_name] = learner_addrs
                to_change_key_values["learner_proxy_num"] = learner_proxy_num
                to_change_key_values["learner_addrs"] = old_learner_address_map

            # 修改配置文件内容落地
            if actor_addrs or learner_addrs:
                if KaiwuDRLDefine.AISRV_FRAMEWORK_KAIWUDRL == CONFIG.aisrv_framework:
                    self.save_to_file(KaiwuDRLDefine.SERVER_AISRV, to_change_key_values)

                CONFIG.write_to_config(to_change_key_values)
                CONFIG.save_to_file(KaiwuDRLDefine.SERVER_AISRV, to_change_key_values)

                self.logger.info(f"AiServer {KaiwuDRLDefine.SERVER_AISRV} CONFIG save_to_file success")

        else:
            if not self_play_actor_address and not self_play_learner_address and not self_play_old_actor_address:
                return

            if self_play_actor_address:
                self_play_actor_proxy_num = len(self_play_actor_address)
                old_actor_address_map[CONFIG.self_play_policy] = self_play_actor_address
                to_change_key_values["self_play_actor_proxy_num"] = self_play_actor_proxy_num

            if self_play_old_actor_address:
                self_play_old_actor_proxy_num = len(self_play_old_actor_address)
                old_actor_address_map[CONFIG.self_play_old_policy] = self_play_old_actor_address
                to_change_key_values["self_play_old_actor_proxy_num"] = self_play_old_actor_proxy_num

            to_change_key_values["actor_addrs"] = old_actor_address_map

            if self_play_learner_address:
                self_play_learner_proxy_num = len(self_play_learner_address)
                CONFIG.self_play_learner_proxy_num = self_play_learner_proxy_num
                old_learner_address_map[CONFIG.self_play_policy] = self_play_learner_address
                to_change_key_values["self_play_learner_proxy_num"] = self_play_learner_proxy_num

            if self_play_old_learner_address:
                self_play_old_learner_proxy_num = len(self_play_old_learner_address)
                CONFIG.self_play_old_learner_proxy_num = self_play_old_learner_proxy_num
                old_learner_address_map[CONFIG.self_play_old_policy] = self_play_old_learner_address
                to_change_key_values["self_play_old_learner_proxy_num"] = self_play_old_learner_proxy_num

            to_change_key_values["learner_addrs"] = old_learner_address_map

            # 修改配置文件内容落地
            if (
                self_play_actor_address
                or self_play_learner_address
                or self_play_old_actor_address
                or self_play_old_learner_address
            ):
                if KaiwuDRLDefine.AISRV_FRAMEWORK_KAIWUDRL == CONFIG.aisrv_framework:
                    self.save_to_file(KaiwuDRLDefine.SERVER_AISRV, to_change_key_values)

                CONFIG.write_to_config(to_change_key_values)
                CONFIG.save_to_file(KaiwuDRLDefine.SERVER_AISRV, to_change_key_values)

                self.logger.info(f"AiServer {KaiwuDRLDefine.SERVER_AISRV} CONFIG save_to_file success")

    def run(self) -> None:
        if not self.before_run():
            self.logger.error(f"AiServer before_run failed, so return")
            return

        while True:
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
                    f"AiServer failed to run {self.name} . exit. Error is: {e}, "
                    f"traceback.print_exc() is {traceback.format_exc()}"
                )

    def run_once(self) -> None:
        # 步骤1, 启动定时器操作, 定时器里执行记录统计信息
        schedule.run_pending()

        # 步骤2, 执行on-policy逻辑
        self.aisrv_on_policy_process()

    # 框架运行前创建必要的文件目录
    def make_dirs(self):
        make_single_dir(CONFIG.log_dir)

    def start_aisrv_handler(self):
        """

        aisrv在启动时, 从alloc进程获取actor和learner的分配IP地址; 如果不从alloc访问默认是采用配置项, 配置项里会设置具体的地址
            在单个容器里是127.0.0.1
            在多个容器里是需要用户自己设置
        1. actor, learner的分配IP地址
        2. kaiwu_env的分配IP地址
        """
        kaiwu_env_default_address = CONFIG.kaiwu_env_default_address
        if kaiwu_env_default_address:
            kaiwu_env_address = kaiwu_env_default_address.split(",")
        else:
            kaiwu_env_address = [f"{KaiwuDRLDefine.LOCAL_HOST_IP}:{CONFIG.kaiwu_env_svr_port}"]
        if int(CONFIG.use_alloc):
            registry_result = self.aisrv_registry_to_alloc()
            if not registry_result:
                self.logger.error(f"AiServer aisrv_registry_to_alloc failed")
            else:
                # 如果需要从alloc服务获取kaiwu_env地址则获取
                if CONFIG.get_kaiwu_env_by_alloc:
                    kaiwu_env_address = self.get_kaiwu_env_ip_from_alloc()

        if not kaiwu_env_address:
            self.logger.error(f"AiServer fail to get kaiwu_env address")
            return

        # aisrv主进程给aisrv_handler进程传递参数
        self.simu_ctx.kaiwu_rl_helper_pause_count_lock = self.kaiwu_rl_helper_pause_count_lock
        self.simu_ctx.kaiwu_rl_helper_pause_count = self.kaiwu_rl_helper_pause_count
        self.simu_ctx.kaiwu_rl_helper_continue_count_lock = self.kaiwu_rl_helper_continue_count_lock
        self.simu_ctx.kaiwu_rl_helper_continue_count = self.kaiwu_rl_helper_continue_count
        self.simu_ctx.get_active_handlers_model_version_value = self.get_active_handlers_model_version_value.__get__(
            self
        )
        self.simu_ctx.get_kaiwu_rl_helper_continue_value = self.get_kaiwu_rl_helper_continue_value.__get__(self)
        self.simu_ctx.get_have_set_kaiwu_rl_helper_continue = self.get_have_set_kaiwu_rl_helper_continue.__get__(self)
        self.simu_ctx.get_have_set_kaiwu_rl_helper_pause = self.get_have_set_kaiwu_rl_helper_pause.__get__(self)
        self.simu_ctx.set_have_set_kaiwu_rl_helper_pause = self.set_have_set_kaiwu_rl_helper_pause.__get__(self)
        self.simu_ctx.set_active_handlers_alive_count_value = self.set_active_handlers_alive_count_value.__get__(self)
        self.simu_ctx.set_have_set_kaiwu_rl_helper_continue = self.set_have_set_kaiwu_rl_helper_continue.__get__(self)

        # 针对返回来的kaiwu_env的IP列表, 每个IP启动单个进程
        for address in kaiwu_env_address:
            if address:
                handler = AiSrvHandle(address, self.simu_ctx)
                handler.start()
                self.logger.info(f"AiServer AiSrvHandle with address: {address} start")

    def before_run(self):
        self.make_dirs()

        # 设置Context
        self.simu_ctx = Context()
        # aisrv handler进程使用
        self.simu_ctx.slots = Slots(int(CONFIG.max_tcp_count), int(CONFIG.max_queue_len))

        # aisrv进程启动时, 从七彩石获取配置, 然后将该七彩石的操作句柄传给对应的子进程
        if int(CONFIG.use_rainbow):
            from kaiwudrl.common.utils.rainbow_wrapper import RainbowWrapper

            rainbow_wrapper = RainbowWrapper(self.logger)
            # 在本次对局开始前, aisrv看下参数修改情况
            rainbow_wrapper.rainbow_activate_single_process(KaiwuDRLDefine.SERVER_MAIN, self.logger)
            rainbow_wrapper.rainbow_activate_single_process(CONFIG.svr_name, self.logger)

            self.simu_ctx.rainbow_wrapper = rainbow_wrapper

        # aisrv启动时获取actor和learner地址
        if int(CONFIG.use_alloc):
            # alloc 工具类, aisrv上与alloc交互操作
            self.alloc_util = AllocUtils(self.logger)
            self.aisrv_registry_to_alloc()
            self.get_actor_learner_ip_from_alloc()

        # 无论从七彩石或者其他地方配置完成的配置文件后再开始检测配置文件的有效性
        if not self.check_param():
            self.logger.error(f"AiServer check_param failed, so return")
            return False

        """
        如果在小规模场景下, 因为model_file_sync进程只需要启动1个, 而actor_proxy_local进程是多个的, 故这里需要采用下面步骤:
        1. model_file_sync进程先启动
        2. 将model_file_sync进程的对象句柄传入到actor_proxy_local进程里进行使用
        又由于会对不同的policy进行AsyncBuilder, 故只有将对model_file_sync的进程启动放在AsyncBuilder调用之前进行
        """
        if CONFIG.predict_local_or_remote == KaiwuDRLDefine.PREDICT_AS_LOCAL:
            model_file_sync_wrapper = ModelFileSyncWrapper()
            model_file_sync_wrapper.init()

            self.simu_ctx.model_file_sync_wrapper = model_file_sync_wrapper

        """
        实例配置如下
        {
            "hero": {
                "run_handler": "app.gym.gym_run_handler.GymRunHandler",
                "rl_helper": "app.gorge_walk.environment.gorge_walk_rl_helper.GorgeWalkRLHelper",
                "policies": {
                "train_one": {
                    "policy_builder" : "kaiwudrl.server.aisrv.async_policy.AsyncBuilder",
                    "algo": "ppo",
                    "state": "app.gym.gym_proto.GymState",
                    "action": "app.gym.gym_proto.GymAction",
                    "reward": "app.gym.gym_proto.GymReward",
                    "actor_network": "app.gym.gym_network.GymDeepNetwork",
                    "learner_network": "app.gym.gym_network.GymDeepNetwork",
                    "reward_shaper": "app.gym.gym_reward_shaper.GymRewardShaper"
                    }
                }
            }
        }
        """
        # 配置相关的传递
        try:
            policies_builder = {}
            policies_conf = AppConf[CONFIG.app].policies
            for policy_name, policy_conf in policies_conf.items():
                policies_builder[policy_name] = policy_conf.policy_builder(policy_name, self.simu_ctx)

            self.simu_ctx.policies_builder = policies_builder

            self.simu_ctx.kaiwu_rl_helper = AppConf[CONFIG.app].rl_helper

        except Exception as e:
            self.logger.exception(
                f"AiSrv server start exception: {str(e)}, traceback.print_exc() is {traceback.format_exc()}"
            )
            return False

        # 启动独立的进程, 负责actor与alloc交互
        if int(CONFIG.use_alloc):
            self.alloc_proxy = AllocProxy()
            self.alloc_proxy.start()

        # 启动独立的进程, 负责actor与普罗米修斯交互
        if int(CONFIG.use_prometheus):
            self.monitor_proxy = MonitorProxy()
            self.monitor_proxy.start()
            self.simu_ctx.monitor_proxy = self.monitor_proxy

        """
        设置了aisrv自动更新actor和learner后, 就设置按时执行
        """
        if CONFIG.actor_learner_expansion:
            set_schedule_event(
                int(CONFIG.alloc_process_per_seconds),
                self.aisrv_with_new_actor_learner_change,
            )

        set_schedule_event(CONFIG.prometheus_stat_per_minutes, self.aisrv_main_process_stat)

        self.process_run_count = 0

        # 便于Aisrv和AiHandler进程之间通信, 因为只是需要model_version版本号, 采用Value数据结构
        if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
            self.active_handlers_model_version = multiprocessing.Value("i", -1)
            self.active_handlers_model_version_lock = multiprocessing.RLock()

        if (
            CONFIG.on_policy_by_way == KaiwuDRLDefine.ALGORITHM_ON_POLICY_WAY_EPISODE
            or CONFIG.on_policy_by_way == KaiwuDRLDefine.ALGORITHM_ON_POLICY_WAY_STEP
        ):
            # 便于Aisrv主进程和AiHandler进程之间通信, 记录当前处于已经被pause的kaiwu_rl_helper线程数量, 采用Value数据结构
            self.kaiwu_rl_helper_pause_count = multiprocessing.Value("i", 0)
            self.kaiwu_rl_helper_pause_count_lock = multiprocessing.RLock()

            # 便于Aisrv主进程和AiHandler进程之间通信, 记录当前处于已经被continue的kaiwu_rl_helper线程数量, 采用Value数据结构
            self.kaiwu_rl_helper_continue_count = multiprocessing.Value("i", 0)
            self.kaiwu_rl_helper_continue_count_lock = multiprocessing.RLock()

            # 便于Aisrv主进程和AiHandler进程之间通信, 表示可以继续暂停掉的线程, 布尔类型, 采用Value数据结构
            self.kaiwu_rl_helper_continue_value = multiprocessing.Value("b", False)
            self.kaiwu_rl_helper_continue_value_lock = multiprocessing.RLock()

            """
            因为单局/单帧会多次continue和pause, 故单个handler需要单独控制的, 故采用Array数据结构, 不能采用Value数据结构
            """
            # 便于Aisrv主进程和AiHandler进程之间通信, 用于多次执行continue操作, 布尔数组类型, 采用Array数据结构
            self.have_set_kaiwu_rl_helper_continue = multiprocessing.Array("b", [False] * CONFIG.max_tcp_count)
            self.have_set_kaiwu_rl_helper_continue_lock = multiprocessing.RLock()

            # 便于Aisrv主进程和AiHandler进程之间通信, 用于多次执行pause操作, 布尔数组类型, 采用Array数据结构
            self.have_set_kaiwu_rl_helper_pause = multiprocessing.Array("b", [False] * CONFIG.max_tcp_count)
            self.have_set_kaiwu_rl_helper_pause_lock = multiprocessing.RLock()

            # 便于Aisrv主进程和AiHandler进程之间通信, 记录当前活着的handler数量, 采用Value数据结构
            self.active_handlers_alive_count = multiprocessing.Value("i", 0)
            self.active_handlers_alive_count_lock = multiprocessing.RLock()

        # 在on-policy的情况下需要启动zmq server, aisrv为server, learner为client
        if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
            zmq_server_port = int(CONFIG.aisrv_server_port) + 100
            self.zmq_server = ZmqServer(CONFIG.aisrv_ip_address, zmq_server_port, duplex=True)
            self.zmq_server.bind()
            self.logger.info(f"AiSrv on-policy aisrv bind at {zmq_server_port} for learner")

            # 统计信息
            self.all_kaiwu_rl_helper_pause_error_count = 0
            self.all_kaiwu_rl_helper_pause_success_count = 0
            self.all_kaiwu_rl_helper_continue_error_count = 0
            self.all_kaiwu_rl_helper_continue_success_count = 0
            self.aisrv_change_model_version_error_count = 0
            self.aisrv_change_model_version_success_count = 0

            # 记录learner所在的client_id
            self.client_id = None

            # learner需要朝aisrv发送了心跳请求后才能得到client_id进行通信
            self.learner_have_send_heartbeat_request_success = False

            # 如果是aisrv按照单帧或者单局的进行, 则表示本次需要推动learner的on-policy流程, 执行完成后就设置为False, 进行下一轮
            if (
                CONFIG.on_policy_by_way == KaiwuDRLDefine.ALGORITHM_ON_POLICY_WAY_EPISODE
                or CONFIG.on_policy_by_way == KaiwuDRLDefine.ALGORITHM_ON_POLICY_WAY_STEP
            ):
                self.aisrv_ask_learner_to_execute_on_policy = False

        # 启动aisrv_handler进程
        self.start_aisrv_handler()

        # 在before run最后打印启动成功日志
        self.logger.info(
            f"AiServer is start success at {CONFIG.aisrv_ip_address}:{CONFIG.aisrv_server_port}, "
            f"pid is {os.getpid()}, run_mode is {CONFIG.run_mode}, "
            f"self_play is {CONFIG.self_play}"
        )
        return True

    # 在kaiwu_rl_helper_pause_count的大小
    def get_kaiwu_rl_helper_pause_count(self):
        kaiwu_rl_helper_pause_count = 0
        with self.kaiwu_rl_helper_pause_count_lock:
            kaiwu_rl_helper_pause_count = self.kaiwu_rl_helper_pause_count.value

        return kaiwu_rl_helper_pause_count

    # 获取self.kaiwu_rl_helper_continue_value的值, 便于handler进程continue kaiwu_rl_helper线程
    def get_kaiwu_rl_helper_continue_value(self):
        kaiwu_rl_helper_continue_value = False
        with self.kaiwu_rl_helper_continue_value_lock:
            kaiwu_rl_helper_continue_value = self.kaiwu_rl_helper_continue_value.value

        return kaiwu_rl_helper_continue_value

    # 获取self.have_set_kaiwu_rl_helper_continue的值, 便于单个aisrv handler处理
    def get_have_set_kaiwu_rl_helper_continue(self, slot_id):
        have_set_kaiwu_rl_helper_continue = False
        with self.have_set_kaiwu_rl_helper_continue_lock:
            have_set_kaiwu_rl_helper_continue = self.have_set_kaiwu_rl_helper_continue[slot_id]

        return have_set_kaiwu_rl_helper_continue

    # 设置self.have_set_kaiwu_rl_helper_continue的值, 便于单个aisrv handler处理
    def set_have_set_kaiwu_rl_helper_continue(self, slot_id, value):
        with self.have_set_kaiwu_rl_helper_continue_lock:
            self.have_set_kaiwu_rl_helper_continue[slot_id] = value

    # 获取self.have_set_kaiwu_rl_helper_pause的值, 便于单个aisrv handler处理
    def get_have_set_kaiwu_rl_helper_pause(self, slot_id):
        have_set_kaiwu_rl_helper_pause = False
        with self.have_set_kaiwu_rl_helper_pause_lock:
            have_set_kaiwu_rl_helper_pause = self.have_set_kaiwu_rl_helper_pause[slot_id]

        return have_set_kaiwu_rl_helper_pause

    # 设置self.have_set_kaiwu_rl_helper_pause的值, 便于单个aisrv handler处理
    def set_have_set_kaiwu_rl_helper_pause(self, slot_id, value):
        with self.have_set_kaiwu_rl_helper_pause_lock:
            self.have_set_kaiwu_rl_helper_pause[slot_id] = value

    # 在kaiwu_rl_helper_continue_count的大小
    def get_kaiwu_rl_helper_continue_count(self):
        kaiwu_rl_helper_continue_count = 0
        with self.kaiwu_rl_helper_continue_count_lock:
            kaiwu_rl_helper_continue_count = self.kaiwu_rl_helper_continue_count.value

        return kaiwu_rl_helper_continue_count

    # 获取self.active_handlers_model_version的值, 便于handler进程model_version更改
    def get_active_handlers_model_version_value(self):
        active_handlers_model_version_value = 0
        with self.active_handlers_model_version_lock:
            active_handlers_model_version_value = self.active_handlers_model_version.value

        return active_handlers_model_version_value

    # 获取self.active_handlers_alive_count的值, 即活着的handler进程
    def get_active_handlers_alive_count(self):
        active_handlers_alive_count_value = 0
        with self.active_handlers_alive_count_lock:
            active_handlers_alive_count_value = self.active_handlers_alive_count.value

        return active_handlers_alive_count_value

    # 设置self.active_handlers_alive_count的值, 主要是单个handler进程退出或者增加使用
    def set_active_handlers_alive_count_value(self, value):
        with self.active_handlers_alive_count_lock:
            self.active_handlers_alive_count.value += value

    # on-policy情况下按照单帧/单局的逻辑
    def on_policy_by_episode(self):
        self.on_policy_by_episode_detail()

    def on_policy_by_episode_detail(self):
        """
        如果是on-policy时, 则流程如下:
        1. 获取所有的kaiwu_rl_helper的暂停信息, 判断依据:self.kaiwu_rl_helper_pause_count的大小等于active_handlers_alive_count大小
        2. 如果1满足, 则朝learner传递开始走on-policy的信息
        3. 如果2满足, 则等待到超时时间后即告警, 后期做容灾处理
        """

        # 如果learner还没有发送心跳请求, 则learner与aisrv之间无法进行通信, 故不走该逻辑
        if not self.learner_have_send_heartbeat_request_success:
            return

        # 如果当前的self.aisrv_ask_learner_to_execute_on_policy已经为True, 说明还没有走完aisrv让learner执行on-policy流程, 故本次就提前返回
        if self.aisrv_ask_learner_to_execute_on_policy:
            return

        """
        在超时时间内, 只要aisrv收到了1个kaiwu_rl_helper发送的on-policy流程开始的请求, 那么就按照最大超时时间收集所有的kaiwu_rl_helper的请求
        并且只是增加1次, 否则无法跳出循环
        """
        end_time = time.time() + CONFIG.on_policy_timeout_seconds
        any_kaiwu_rl_helper_pause_success = False
        update_end_time = False
        kaiwu_rl_helper_pause_count = 0
        active_handle_count = 0
        while time.time() < end_time:
            # 新增/减少handler, 新增/减少kaiwu_rl_helper线程是动态的
            kaiwu_rl_helper_pause_count = self.get_kaiwu_rl_helper_pause_count()
            active_handle_count = self.get_active_handlers_alive_count()
            if kaiwu_rl_helper_pause_count > 0:
                any_kaiwu_rl_helper_pause_success = True
            else:
                # 减少CPU争用
                time.sleep(CONFIG.idle_sleep_second)

            # 达到一定比例即跳出循环
            if any_kaiwu_rl_helper_pause_success and (
                kaiwu_rl_helper_pause_count / active_handle_count >= CONFIG.on_policy_quantity_ratio
            ):
                break

            if any_kaiwu_rl_helper_pause_success and not update_end_time:
                end_time = time.time() + CONFIG.on_policy_timeout_seconds
                update_end_time = True

        # 超时后即告警处理
        if any_kaiwu_rl_helper_pause_success:
            if kaiwu_rl_helper_pause_count / active_handle_count < CONFIG.on_policy_quantity_ratio:
                # 设置本次需要learner执行on-policy逻辑的标志位
                self.aisrv_ask_learner_to_execute_on_policy = False

                self.all_kaiwu_rl_helper_pause_error_count += 1
                self.logger.error(
                    f"AiSrv on_policy_by_episode kaiwu_rl_helper_pause_count {kaiwu_rl_helper_pause_count} / "
                    f"active_handle_count {active_handle_count} < {CONFIG.on_policy_quantity_ratio}, so aisrv not "
                    f"ask learner to execute on-policy process this time, try next time"
                )
                return

            else:
                # 设置本次需要learner执行on-policy逻辑的标志位
                self.aisrv_ask_learner_to_execute_on_policy = True
                self.all_kaiwu_rl_helper_pause_success_count += 1
                self.logger.info(
                    f"AiSrv on_policy_by_episode kaiwu_rl_helper_pause_count {kaiwu_rl_helper_pause_count} / "
                    f"active_handle_count {active_handle_count} >= {CONFIG.on_policy_quantity_ratio}, so "
                    f"self.aisrv_ask_learner_to_execute_on_policy is True"
                )
        else:
            # 设置本次需要learner执行on-policy逻辑的标志位
            self.aisrv_ask_learner_to_execute_on_policy = False

        # 如果需要aisrv发起learner的on-policy流程
        self.on_policy_aisrv_ask_learner_to_execute_on_policy()

    def on_policy_aisrv_ask_learner_to_execute_on_policy(self):
        # 如果需要aisrv发起learner的on-policy流程
        if self.aisrv_ask_learner_to_execute_on_policy:
            message_type_value = KaiwuDRLDefine.ON_POLICY_MESSAGE_ASK_LEARNER_TO_EXECUTE_ON_POLICY_PROCESS_REQUEST
            message = {KaiwuDRLDefine.MESSAGE_TYPE: message_type_value}

            # 便于以后扩展, 采用map形式
            value = self.aisrv_ask_learner_to_execute_on_policy
            send_data = {KaiwuDRLDefine.ON_POLICY_MESSAGE_ASK_LEARNER_TO_EXECUTE_ON_POLICY_PROCESS_REQUEST: value}
            message[KaiwuDRLDefine.MESSAGE_VALUE] = send_data

            # aisrv返回给learner确认信息
            self.zmq_server.send(str(self.client_id), message, binary=False)

            self.logger.info(
                f"AiSrv learner ask aisrv {self.client_id} msg_type: {message[KaiwuDRLDefine.MESSAGE_TYPE]} success"
            )

    # on-policy情况下, aisrv与learner之间的通信
    def on_policy_aisrv_communication_with_learner(self):
        # 在aisrv发起的on-policy流程主要是针对每局或者每帧, 此时也需要收集从learner发给aisrv的model_version修改信息
        try:
            self.client_id, message = self.zmq_server.recv(block=False, binary=False)
            if message:
                """
                需要区分消息类型
                1. 如果是learner的heartbeat请求, 主要是为获取client_id, 增加是否需要开启on-policy流程, 需要回包给learner
                2. 如果是learner的model_version请求, 主要是更新model_version, 需要回包给learner
                3. 待扩展
                """
                if message[KaiwuDRLDefine.MESSAGE_TYPE] == KaiwuDRLDefine.ON_POLICY_MESSAGE_HEARTBEAT_REQUEST:
                    message[KaiwuDRLDefine.MESSAGE_TYPE] = KaiwuDRLDefine.ON_POLICY_MESSAGE_HEARTBEAT_RESPONSE

                    message[KaiwuDRLDefine.MESSAGE_VALUE] = True

                    self.learner_have_send_heartbeat_request_success = True

                    # aisrv返回给learner确认信息
                    self.zmq_server.send(str(self.client_id), message, binary=False)

                    self.logger.debug(
                        f"AiSrv learner ask aisrv {self.client_id} "
                        f"msg_type: {message[KaiwuDRLDefine.MESSAGE_TYPE]} success"
                    )

                elif (
                    message[KaiwuDRLDefine.MESSAGE_TYPE]
                    == KaiwuDRLDefine.ON_POLICY_MESSAGE_MODEL_VERSION_CHANGE_REQUEST
                ):
                    model_version = message[KaiwuDRLDefine.MESSAGE_VALUE]

                    message[
                        KaiwuDRLDefine.MESSAGE_TYPE
                    ] = KaiwuDRLDefine.ON_POLICY_MESSAGE_MODEL_VERSION_CHANGE_RESPONSE

                    aisrv_process_model_version_success = self.aisrv_process_when_model_version_change(model_version)
                    if aisrv_process_model_version_success:
                        self.aisrv_change_model_version_success_count += 1
                    else:
                        self.aisrv_change_model_version_error_count += 1

                    message[KaiwuDRLDefine.MESSAGE_VALUE] = aisrv_process_model_version_success

                    # aisrv返回给learner确认信息
                    self.zmq_server.send(str(self.client_id), message, binary=False)

                    self.logger.info(
                        f"AiSrv learner ask aisrv {self.client_id} "
                        f"msg_type: {message[KaiwuDRLDefine.MESSAGE_TYPE]} success"
                    )

                else:
                    self.logger.error(
                        f"AiSrv learner ask aisrv not support msg_type: {message[KaiwuDRLDefine.MESSAGE_TYPE]}"
                    )
                    return

        except Exception as e:
            pass

    def aisrv_process_when_model_version_change(self, model_version):
        """
        learner让aisrv走on-policy流程, aisrv其具体的有下面操作:
        1. 更新kaiwu_rl_helper的model_version
        2. aisrv主进程设置继续kaiwu_rl_helper线程标志位为True
        3. aisrv的handler进程看见2中的标准位, 继续kaiwu_rl_helper线程
        4. aisrv主进程收集所有的handler进程标志位
        4.1 如果不等于则超时处理
        4.2 如果等于则执行5
        5. aisrv主进程设置继续kaiwu_rl_helper线程标志位为False
        6. aisrv主进程将self.kaiwu_rl_helper_continue_count全部赋值为False
        7. 给learner回响应包代表本次流程OK
        8. 初始化部分指标, 下个周期再进入

        返回True表示执行成功, 返回False表示执行失败, 需要做容错处理
        """

        aisrv_process_model_version_success = True

        # 设置handler_version
        with self.active_handlers_model_version_lock:
            self.active_handlers_model_version.value = model_version

        # 按照单帧/单局需要走的逻辑
        if (
            CONFIG.on_policy_by_way == KaiwuDRLDefine.ALGORITHM_ON_POLICY_WAY_EPISODE
            or CONFIG.on_policy_by_way == KaiwuDRLDefine.ALGORITHM_ON_POLICY_WAY_STEP
        ):
            # 设置kaiwu_rl_helper线程继续的标志位
            with self.kaiwu_rl_helper_continue_value_lock:
                self.kaiwu_rl_helper_continue_value.value = True

            retry_count = 0
            kaiwu_rl_helper_continue_count = 0
            active_handlers_alive_count = 0
            while retry_count < int(CONFIG.on_policy_error_retry_count):
                # 需要实时计算值
                kaiwu_rl_helper_continue_count = self.get_kaiwu_rl_helper_continue_count()
                active_handlers_alive_count = self.get_active_handlers_alive_count()
                if kaiwu_rl_helper_continue_count / active_handlers_alive_count >= CONFIG.on_policy_quantity_ratio:
                    break
                else:
                    # 减少CPU争用
                    time.sleep(CONFIG.idle_sleep_second)

                retry_count += 1

            # 如果超过了最大重试次数, 则计入监控, 后期做容错处理
            if retry_count >= int(CONFIG.on_policy_error_retry_count):
                self.logger.error(
                    f"AiSrv main process not recv all handler continue kaiwu_rl_helper response, "
                    f"kaiwu_rl_helper_continue_count {kaiwu_rl_helper_continue_count}, "
                    f"active_handlers_alive_count {active_handlers_alive_count}"
                )
                aisrv_process_model_version_success = False
                self.all_kaiwu_rl_helper_continue_error_count += 1
            else:
                self.logger.info(f"AiSrv main process recv all handler continue kaiwu_rl_helper response")
                self.all_kaiwu_rl_helper_continue_success_count += 1

            # 无论本次执行成功与否, 下面变量都reset下, 便于下个周期进行处理, 否则就出现aisrv的死锁状态
            self.aisrv_ask_learner_to_execute_on_policy = False
            with self.kaiwu_rl_helper_pause_count_lock:
                self.kaiwu_rl_helper_pause_count.value = 0

            with self.kaiwu_rl_helper_continue_value_lock:
                self.kaiwu_rl_helper_continue_value.value = False

            with self.kaiwu_rl_helper_continue_count_lock:
                self.kaiwu_rl_helper_continue_count.value = 0

            with self.have_set_kaiwu_rl_helper_continue_lock:
                self.have_set_kaiwu_rl_helper_continue[:] = [False] * len(self.have_set_kaiwu_rl_helper_continue)

            with self.have_set_kaiwu_rl_helper_pause_lock:
                self.have_set_kaiwu_rl_helper_pause[:] = [False] * len(self.have_set_kaiwu_rl_helper_pause)

        return aisrv_process_model_version_success

    # aisrv启动的learner的on-policy流程
    def aisrv_on_policy_process(self):
        if CONFIG.algorithm_on_policy_or_off_policy != KaiwuDRLDefine.ALGORITHM_ON_POLICY:
            return

        """
        在on-policy的while True循环里, 主要流程:
        1. 如果是按照对局或者单帧的, 需要走self.on_policy_by_episode()函数
        2. aisrv与learner之间通信交互
        """
        while True:
            if (
                CONFIG.on_policy_by_way == KaiwuDRLDefine.ALGORITHM_ON_POLICY_WAY_EPISODE
                or CONFIG.on_policy_by_way == KaiwuDRLDefine.ALGORITHM_ON_POLICY_WAY_STEP
            ):
                self.on_policy_by_episode()

            # aisrv与learner之间的通信交互函数
            self.on_policy_aisrv_communication_with_learner()


class AiSrvHandle(multiprocessing.Process):
    __slots__ = (
        "logger",
        "conn",
        "simu_ctx",
        "slots",
        "slot_id",
        "msg_buff",
        "data_queue",
        "kaiwu_rl_helper",
        "min_slot_id",
        "monitor_proxy",
    )

    def __init__(self, kaiwu_env_address, simu_ctx) -> None:
        super(AiSrvHandle, self).__init__()
        self.kaiwu_env_address = kaiwu_env_address
        self.simu_ctx = simu_ctx

        # 建立zmq通信
        """
        self.client_id = get_uuid()
        parts = self.kaiwu_env_address.split(":")
        ip = parts[0]
        port = parts[1]
        self.zmq_client = ZmqClient(str(self.client_id), ip, port)
        self.zmq_client.connect()
        """

        self.simu_ctx.client_address = self.kaiwu_env_address

        self.simu_ctx.exit_flag = multiprocessing.Value("b", False)

        # slot_id
        self.slots = self.simu_ctx.slots
        self.slot_id = self.slots.get_slot()
        self.simu_ctx.slot_id = self.slot_id

        # 七彩石的操作句柄由主进程传递
        if int(CONFIG.use_rainbow):
            self.rainbow_wrapper = self.simu_ctx.rainbow_wrapper

        # 设置aisrv上对客户端的消息buff, 匹配速度
        self.msg_buff = MsgBuff(self.simu_ctx)
        self.simu_ctx.msg_buff = self.msg_buff

        # 负责统计kaiwu_rl_helper中产生的对局数据，线程和主进程间只用数据dequeue即可，减少cpu消耗
        self.data_queue = collections.deque(maxlen=CONFIG.max_queue_len)
        self.simu_ctx.data_queue = self.data_queue

        self.process_run_count = 0

    # 设置每个handler的model_version
    def set_handler_model_version(self, model_version):
        self.kaiwu_rl_helper.from_learner_model_version = model_version
        self.logger.info(f"aisrvhandle on-policy set_handler_model_version success, model_version: {model_version}")

    # aisrv在处理actor和learner的动态扩缩容逻辑
    def aisrv_with_new_actor_learner_change(self):
        if not CONFIG.actor_learner_expansion:
            return

        (
            current_actor_addrs,
            current_learner_addrs,
        ) = self.kaiwu_rl_helper.get_current_actor_learner_address()

        read_from_file_content = CONFIG.read_from_file(CONFIG.svr_name, ["actor_addrs", "learner_addrs"])

        # 本次读取的文件内容错误, 则跳过本次处理下次再进行处理
        try:
            new_actor_addrs = read_from_file_content["actor_addrs"][CONFIG.policy_name]
            new_learner_addrs = read_from_file_content["learner_addrs"][CONFIG.policy_name]
        except Exception as e:
            self.logger.info(f"aisrvhandle load actor address and learner address err, {str(e)}")
            return

        self.aisrv_with_different_actor_learner(
            current_actor_addrs,
            new_actor_addrs,
            current_learner_addrs,
            new_learner_addrs,
        )

    # actor和learner的IP区别判断, 采用2个参数进行返回
    def check_actor_ip_and_learner_ip_change(
        self, actor_address, old_actor_address, learner_address, old_learner_addrs
    ):
        actor_ip_change = False
        learner_ip_change = False

        if not actor_address and not learner_address:
            return actor_ip_change, learner_ip_change

        if actor_address:
            if not is_list_eq(actor_address, old_actor_address):
                actor_ip_change = True

        if learner_address:
            if not is_list_eq(learner_address, old_learner_addrs):
                learner_ip_change = True

        return actor_ip_change, learner_ip_change

    def aisrv_with_different_actor_learner(
        self,
        current_actor_addrs,
        new_actor_addrs,
        current_learner_addrs,
        new_learner_addrs,
    ):
        actor_ip_change, learner_ip_change = self.check_actor_ip_and_learner_ip_change(
            new_actor_addrs,
            current_actor_addrs,
            new_learner_addrs,
            current_learner_addrs,
        )

        if not actor_ip_change and not learner_ip_change:
            return

        # actor地址有变化
        if actor_ip_change:
            list_A_have_B_not_have, list_B_have_A_not_have = list_diff(current_actor_addrs, new_actor_addrs)
            if list_A_have_B_not_have:
                # 新的有, 但是旧的没有, AsyncBuilder新增actor_proxy
                actor_add_result = self.kaiwu_rl_helper.kaiwu_rl_helper_change_actor_learner_ip(
                    KaiwuDRLDefine.PROCESS_ADD, list_A_have_B_not_have, None, None
                )

            if list_B_have_A_not_have:
                # 新的没有, 但是旧的有, AsyncBuilder减少actor_ip
                actor_reduce_result = self.kaiwu_rl_helper.kaiwu_rl_helper_change_actor_learner_ip(
                    KaiwuDRLDefine.PROCESS_REDUCE,
                    list_B_have_A_not_have,
                    None,
                    None,
                )

        # learner地址有变化
        if learner_ip_change:
            list_A_have_B_not_have, list_B_have_A_not_have = list_diff(new_learner_addrs, current_learner_addrs)
            if list_A_have_B_not_have:
                # 新的有, 但是旧的没有, AsyncBuilder新增learner_proxy
                learner_add_result = self.kaiwu_rl_helper.kaiwu_rl_helper_change_actor_learner_ip(
                    None, None, KaiwuDRLDefine.PROCESS_ADD, list_A_have_B_not_have
                )

            if list_B_have_A_not_have:
                # 新的没有, 但是旧的有, AsyncBuilder减少learner_ip
                learner_reduce_result = self.kaiwu_rl_helper.kaiwu_rl_helper_change_actor_learner_ip(
                    None,
                    None,
                    KaiwuDRLDefine.PROCESS_REDUCE,
                    list_B_have_A_not_have,
                )

        # 修改配置文件内容落地
        if actor_add_result and actor_reduce_result and learner_add_result and learner_reduce_result:
            self.logger.info("aisrvhandle aisrv_with_different_actor_learner change finish sucess")

    def before_run(self):
        self.logger = KaiwuLogger()
        self.current_pid = os.getpid()
        self.logger.setLoggerFormat(
            (
                f"/{CONFIG.svr_name}/aisrv_handle_pid{self.current_pid}_log_"
                f"{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log"
            ),
            "aisrvhandle",
        )

        """
         aisrv下每1个客户端启动1个KaiWuRLHelper对象, 封装了强化学习流程
         1. 如果是主循环的内容在业务侧, 调用self.kaiwu_rl_helper = self.simu_ctx.kaiwu_rl_helper
         2. 如果是主循环的内容在框架侧, 调用self.kaiwu_rl_helper = KaiWuRLHelper(self.simu_ctx)
         """
        self.kaiwu_rl_helper = self.simu_ctx.kaiwu_rl_helper(self.simu_ctx)

        # self.kaiwu_rl_helper = KaiWuRLHelper(self.simu_ctx)

        self.kaiwu_rl_helper.daemon = True
        self.logger.info(f"aisrvhandle use kaiwu_rl_helper: {self.kaiwu_rl_helper}")

        self.min_slot_id, _ = self.slots.get_min_max_slot_id()
        self.logger.info(
            f"aisrvhandle established connect to {self.kaiwu_env_address}, "
            f"slot id is {self.slot_id}, min_slot_id is {self.min_slot_id}"
        )

        (
            current_actor_addrs,
            current_learner_addrs,
        ) = self.kaiwu_rl_helper.get_current_actor_learner_address()
        self.logger.info(
            f"aisrvhandle current_actor_addrs is {current_actor_addrs}, "
            f"current_learner_addrs is {current_learner_addrs}"
        )

        # 启动独立的进程, 负责aisrv与普罗米修斯交互
        if int(CONFIG.use_prometheus):
            self.monitor_proxy = self.simu_ctx.monitor_proxy

            # 传递给kaiwu_rl_helper
            self.kaiwu_rl_helper.set_monitor_proxy(self.monitor_proxy)

        # 支持每局结束前, 动态修改配置文件
        if int(CONFIG.use_rainbow):
            # 在本次对局开始前, aisrv看下参数修改情况
            self.rainbow_wrapper.rainbow_activate_single_process(KaiwuDRLDefine.SERVER_MAIN, self.logger)
            self.rainbow_wrapper.rainbow_activate_single_process(CONFIG.svr_name, self.logger)

        """
        设置了aisrv自动更新actor和learner后, 就设置按时执行
        """
        if CONFIG.actor_learner_expansion:
            set_schedule_event(
                int(CONFIG.alloc_process_per_seconds),
                self.aisrv_with_new_actor_learner_change,
            )

        # 单局开始时
        self.episode_start()

        # 在before run最后打印启动成功日志
        self.logger.info(f"aisrvhandle start success at pid {self.current_pid}")

        # 开启kaiwu_rl_helper线程, 因为在该线程的主循环和其他的进程开始交互, 故放在before_run最后开始
        self.kaiwu_rl_helper.start()

        # 注册SIGTERM信号处理
        register_sigterm_handler(self.handle_sigterm, CONFIG.sigterm_pids_file)

        return True

    # 设置kaiwu_rl_helper继续执行
    def set_kaiwu_rl_helper_continue(self):
        self.kaiwu_rl_helper.process_continue()

    # 更新self.kaiwu_rl_helper_pause_count的值, 主要操作为增加1或者减少1
    def set_kaiwu_rl_helper_should_pause_value(self, value):
        with self.simu_ctx.kaiwu_rl_helper_pause_count_lock:
            self.simu_ctx.kaiwu_rl_helper_pause_count.value += value

    # 更新self.kaiwu_rl_helper_continue_count的值, 主要操作为增加1或者减少1
    def set_kaiwu_rl_helper_should_continue_value(self, value):
        with self.simu_ctx.kaiwu_rl_helper_continue_count_lock:
            self.simu_ctx.kaiwu_rl_helper_continue_count.value += value

    def aisrv_handler_on_policy_process(self):
        """
        aisrv_handler的流程如下:
        1. 如果需要更新kaiwu_rl_helper的model_version, 则更新
        2. 如果是按照步或者对局来执行on-policy的, 则如果需要执行继续kaiwu_rl_helper操作则执行
        3. 如果是按照步或者对局来执行on-policy的, 收集获取到的暂停的kaiwu_rl_helper的数量
        """

        if CONFIG.algorithm_on_policy_or_off_policy != KaiwuDRLDefine.ALGORITHM_ON_POLICY:
            return

        # 记录上次更新的model_version
        self.last_active_handlers_model_version = -1
        # 记录是否已经做过pause操作, 只有做过1次pause操作的才能做continue操作
        self.have_execute_pause = False

        while not self.simu_ctx.exit_flag.value:
            # 如果本次需要更新的model_version和AisrvHandler维护的model_version一致, 则不需要更新, 否则更新
            current_active_model_version = self.simu_ctx.get_active_handlers_model_version_value()
            if self.last_active_handlers_model_version == current_active_model_version:
                # 本次没有model_version更新则短暂休息下, 规避CPU耗用
                time.sleep(CONFIG.idle_sleep_second)
            else:
                # 如果不一致说明learner已经让aisrv更新了新的model_version了, 则代表learner已经走了on-policy的流程
                self.set_handler_model_version(current_active_model_version)
                self.last_active_handlers_model_version = current_active_model_version

            # 按照每帧/每局逻辑下继续进程处理
            if (
                CONFIG.on_policy_by_way == KaiwuDRLDefine.ALGORITHM_ON_POLICY_WAY_EPISODE
                or CONFIG.on_policy_by_way == KaiwuDRLDefine.ALGORITHM_ON_POLICY_WAY_STEP
            ):
                """
                aisrv主进程让handler进程之间的处理, 包括下面的流程:
                1. 如果aisrv主进程让handler进程执行continue操作, 则
                1.1 判断该handler是否做过pause操作, 只有做过pause操作的才能做continue操作, 即顺序是pause->continue->pause->continue ......
                1.2 判断kaiwu_rl_helper是否处于continue状态, 如果是则不用操作, 不是则继续处理
                1.3 判断self.have_set_kaiwu_rl_helper_pause是否为True?
                1.3.1 如果是True, 本次计数已经完成不再计数
                1.3.2 如果是False, 则执行下面的操作
                1.3.2.1 调用self.set_kaiwu_rl_helper_continue()让kaiwu_rl_helper线程继续执行
                1.3.2.2 对aisrv主线程继续的变量进行计数 + 1
                1.3.2.3 设置self.have_set_kaiwu_rl_helper_continue为True
                2. 如果handler进程的kaiwu_rl_helper线程已经执行了pause操作, 则
                2.1 判断kaiwu_rl_helper是否处于暂停状态
                2.2 判断self.have_set_kaiwu_rl_helper_pause是否为True?
                2.2.1 如果是True, 本次计数已经完成不再计数
                2.2.2 如果是False, 本次计数, 并且设置self.have_set_kaiwu_rl_helper_pause为True
                2.3 如果aisrv需要handler进程continue时, 此时让self.have_set_kaiwu_rl_helper_pause为False

                """
                should_continue = self.simu_ctx.get_kaiwu_rl_helper_continue_value()
                should_pause = self.kaiwu_rl_helper.get_process_in_pause_statues()

                # aisrv主进程需要handler进程的kaiwu_rl_helper线程执行continue操作
                if should_continue:
                    if self.have_execute_pause:
                        if not self.simu_ctx.get_have_set_kaiwu_rl_helper_continue(self.slot_id):
                            self.set_kaiwu_rl_helper_continue()
                            self.set_kaiwu_rl_helper_should_continue_value(1)

                            self.simu_ctx.set_have_set_kaiwu_rl_helper_continue(self.slot_id, True)

                # 如果aisrv的handler进程的kaiwu_rl_helper线程处于pause状态
                elif should_pause:
                    if not self.simu_ctx.get_have_set_kaiwu_rl_helper_pause(self.slot_id):
                        self.set_kaiwu_rl_helper_should_pause_value(1)

                        self.simu_ctx.set_have_set_kaiwu_rl_helper_pause(self.slot_id, True)

                        self.have_execute_pause = True

                # 未来扩展
                else:
                    pass

    def run_once(self) -> None:
        # 步骤1, 例行任务
        schedule.run_pending()

        # 步骤2, 执行on-policy逻辑
        if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
            if (
                CONFIG.on_policy_by_way == KaiwuDRLDefine.ALGORITHM_ON_POLICY_WAY_EPISODE
                or CONFIG.on_policy_by_way == KaiwuDRLDefine.ALGORITHM_ON_POLICY_WAY_STEP
            ):
                self.aisrv_handler_on_policy_process()

    def run(self) -> None:
        # before_run
        if not self.before_run():
            self.logger.error(f"aisrvhandle before_run failed, so return")
            return

        # 主循环
        try:
            while not self.simu_ctx.exit_flag.value:
                self.run_once()

                # 短暂sleep, 规避容器里进程CPU使用率100%问题
                self.process_run_count += 1
                if self.process_run_count % CONFIG.idle_sleep_count == 0:
                    time.sleep(CONFIG.idle_sleep_second)

                    # process_run_count置0, 规避溢出
                    self.process_run_count = 0

        except Exception as e:
            self.logger.error(
                f"aisrvhandle failed to handle message {str(e)}, traceback.print_exc() is {traceback.format_exc()}"
            )
            self.simu_ctx.exit_flag.value = True

            self.episode_stop()

            raise e

    def ep_end_req(self):
        builder = flatbuffers.Builder(0)
        ep_end_req = KaiwuMsgHelper.encode_ep_end_req(builder, b"0", 0, b"")
        req = KaiwuMsgHelper.encode_request(builder, 0, ReqMsg.ReqMsg.ep_end_req, ep_end_req)
        builder.Finish(req)
        req_msg = builder.Output()

        self.msg_buff.input_q.put(
            req_msg,
        )
        self.logger.info("aisrvhandle send Ep end frame")

    # 单次对局结束时处理
    def episode_stop(self):
        # 安全退出KaiWuRLHelper
        self.kaiwu_rl_helper.stop()

        # 其他报错同理需要回一个结束包，来结束线程，释放资源
        self.ep_end_req()

        # 回收slot_id
        self.slots.put_slot(self.slot_id)

        # handler的finish方法都会调用
        if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
            if (
                CONFIG.on_policy_by_way == KaiwuDRLDefine.ALGORITHM_ON_POLICY_WAY_EPISODE
                or CONFIG.on_policy_by_way == KaiwuDRLDefine.ALGORITHM_ON_POLICY_WAY_STEP
            ):
                self.simu_ctx.set_active_handlers_alive_count_value(-1)
                self.logger.info(f"aisrvhandle reduce a handler, slot_id: {self.slot_id}")

        self.logger.info("aisrvhandle lost connection from {}", str(self.kaiwu_env_address))

    # 单次对局开始时处理
    def episode_start(self):
        if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
            if (
                CONFIG.on_policy_by_way == KaiwuDRLDefine.ALGORITHM_ON_POLICY_WAY_EPISODE
                or CONFIG.on_policy_by_way == KaiwuDRLDefine.ALGORITHM_ON_POLICY_WAY_STEP
            ):
                # 活着的handler增加1个
                self.simu_ctx.set_active_handlers_alive_count_value(1)
                self.logger.info(f"aisrvhandle add a handler, slot_id: {self.slot_id}")

    def handle_sigterm(self, sig, frame):
        if CONFIG.wrapper_type == KaiwuDRLDefine.WRAPPER_LOCAL:
            self.logger.info(f"aisrv_handle {os.getpid()} is starting to handle the SIGTERM signal.")
            self.kaiwu_rl_helper.handle_sigterm(sig, frame)
        else:
            self.logger.info(f"aisrv_handle not KaiwuDRLDefine.WRAPPER_LOCAL, so {os.getpid()} not handle the SIGTERM signal.")
