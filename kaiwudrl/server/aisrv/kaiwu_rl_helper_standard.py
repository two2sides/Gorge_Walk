#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file kaiwu_rl_helper_standard.py
# @brief
# @author kaiwu
# @date 2023-11-28


import os
import threading
import time
import traceback
import datetime
import numpy as np
import dill
from kaiwudrl.common.utils.common_func import (
    Context,
    get_host_ip,
)
from kaiwudrl.interface.agent_context import AgentContext
from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.config.app_conf import AppConf
from kaiwudrl.common.config.algo_conf import AlgoConf
from kaiwudrl.common.logging.kaiwu_logger import KaiwuLogger
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.common.checkpoint.model_file_common import (
    update_id_list,
    clear_id_list_file,
    update_id_list,
    clear_user_ckpt_dir,
    check_id_valid,
)
import kaiwu_env
import warnings
from kaiwudrl.server.common.load_model_common import LoadModelCommon

from kaiwudrl.common.algorithms.model_wrapper_common import (
    create_standard_model_wrapper,
)


# 实现标准的强化学习训练流程
class KaiWuRLStandardHelper(threading.Thread):
    __slots__ = (
        "policies",
        "simu_ctx",
        "exit_flag",
        "client_address",
        "slot_id",
        "data_queue",
        "client_id",
        "episode_start_time",
        "ep_frame_cnt",
        "agent_ctxs",
        "logger",
        "env",
        "steps",
        "reward_value",
        "use_sample_server",
    )

    def __init__(self, parent_simu_ctx) -> None:
        super().__init__()

        self.policies = {}
        # 根据policy来设置下, 强化学习是AsyncPolicy, 形如train --> AsyncPolicy
        for policy_name, policy_builder in parent_simu_ctx.policies_builder.items():
            self.policies[policy_name] = policy_builder.build()

        # 上下文放在该变量里
        self.simu_ctx = Context(**parent_simu_ctx.__dict__)

        # 是否结束标志位
        self.exit_flag = self.simu_ctx.exit_flag
        # 客户端ID
        self.client_address = self.simu_ctx.client_address
        # policy
        self.simu_ctx.policies = self.policies
        # slot_id
        self.slot_id = self.simu_ctx.slot_id

        # 数据队列
        self.data_queue = self.simu_ctx.data_queue

        # 设置线程名字
        self.setName(f"kaiwu_rl_helper_{self.slot_id}")

        self.client_id = None

        # 下面是episode的统计指标
        self.episode_start_time = 0
        self.ep_frame_cnt = 0

        # 智能体agent的上下文agent_ctxs, 格式为{"agent_id" : agent_ctx}
        self.agent_ctxs = {}

        # 日志模块
        self.logger = KaiwuLogger()
        self.current_pid = os.getpid()
        self.logger.setLoggerFormat(
            f"/{CONFIG.svr_name}/aisrv_kaiwu_rl_helper_pid{self.current_pid}_log_"
            f"{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log",
            CONFIG.svr_name,
        )
        self.logger.info(
            f"kaiwu_rl_helper start at pid {self.current_pid}, "
            f"ppid is {threading.currentThread().ident}, thread id is {self.get_pid()}"
        )

        # 将日志句柄作为参数传递
        self.simu_ctx.logger = self.logger

        # 动作执行步数，用于样本计数
        self.steps = 0

        # reward统计值
        self.reward_value = 0

        # 是否用sample_server进行样本存储
        self.use_sample_server = CONFIG.use_sample_server

        # aisrv发给actor的请求返回给处理该值的model文件版本号
        self.from_actor_model_version = -1

        # learner通知aisrv此时最新的model文件版本号
        self.from_learner_model_version = -1

        # 暂停/继续线程执行
        self.should_pause = False

        # 有多少agent_id
        self.agent_ids = []

        # 业务会在aisrv里上报自定义的监控指标, 故这里增加上, map形式, 由业务自己定义
        self.app_monitor_data = {}

        # 由于某些场景下kaiwu_rl_helper会退出, 此时不确定aisrv_server_standard能否退出, 故将监控对象传递下做最后的上报
        self.monitor_proxy = None

        self.ip = get_host_ip()

        # policy和model对象的map, 为了支持多agent
        self.policy_model_wrapper_maps = {}
        self.current_models = []

        # 统计值
        if CONFIG.wrapper_type == KaiwuDRLDefine.WRAPPER_LOCAL:
            # 进程上报监控时间
            self.last_report_monitor_time = 0

            # 因为pytorch需要用户确保保存最多多少model文件
            self.file_queue = []

    def set_monitor_proxy(self, monitor_proxy):
        self.monitor_proxy = monitor_proxy

    # 获取当前使用的actor和learner列表
    def get_current_actor_learner_address(self):
        actor_addrs, learner_addrs = None, None
        policy_build = self.policies[CONFIG.policy_name]
        if policy_build:
            (
                actor_addrs,
                learner_addrs,
            ) = policy_build.get_current_actor_learner_proxy_list()

        return actor_addrs, learner_addrs

    # 获取当前训练的reward
    def get_current_reward_value(self):
        return self.reward_value

    def kaiwu_rl_helper_change_actor_learner_ip(
        self, actor_add_or_reduce, actor_ips, learner_add_or_reduce, learner_ips
    ):
        """
        修改kaiwu_rl_helper的actor和learner地址
        1. actor_add_or_reduce, 针对actor的增减
        2. actor_ips, actor_ip列表
        3. learner_add_or_reduce, 针对learner的增减
        4. learner_ips, learner_ip列表

        返回的参数:
        1. False, 即本次没有更新, 不能修改old_actor_address和old_learner_address
        2. True, 即本次更新完成, 需要修改old_learner_address和old_learner_address
        """

        if actor_add_or_reduce and not actor_ips:
            return False

        if learner_add_or_reduce and not learner_ips:
            return False

        # 针对当前的policy_name进行处理
        policy = self.policies[CONFIG.policy_name]

        # 下面针对具体的actor和learner的增减进行处理
        if actor_add_or_reduce and actor_ips:
            for actor_ip in actor_ips:
                if KaiwuDRLDefine.PROCESS_ADD == actor_add_or_reduce:
                    policy.add_actor_proxy_list(actor_ip)
                elif KaiwuDRLDefine.PROCESS_REDUCE == actor_add_or_reduce:
                    policy.reduce_actor_proxy_list(actor_ip)
                else:
                    pass

        if learner_add_or_reduce and learner_ips:
            for learner_ip in learner_ips:
                if KaiwuDRLDefine.PROCESS_ADD == learner_add_or_reduce:
                    policy.add_learner_proxy_list(learner_ip)
                elif KaiwuDRLDefine.PROCESS_REDUCE == learner_add_or_reduce:
                    policy.reduce_learner_proxy_list(learner_ip)
                else:
                    pass

        # 操作完成后需要继续线程活动
        self.logger.info(
            f"kaiwu_rl_helper {actor_add_or_reduce} {actor_ips} {learner_add_or_reduce} {learner_ips} "
            f"expansion success"
        )

        return True

    # 返回policies
    def get_policies(self):
        return self.policies

    # 获取线程ID
    def get_pid(self):
        if hasattr(self, "_thread_id"):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id

        return -1

    @property
    def identity(self):
        return f"kaiwu_rl_helper_{self.slot_id}"

        # return "(client_conn_id: %s, client_id: %s)" % (self.client_address, self.client_id or "")

    # 获取是否处于pause状态
    def get_process_in_pause_statues(self):
        return self.should_pause

    # 暂停处理pause
    def process_pause(self):
        self.should_pause = True

    # 继续处理continue
    def process_continue(self):
        self.should_pause = False

    # 目前业务sgame, 每次action
    def gen_expr(self, agent_id, policy_id, extra_info=None):
        agent_ctx = self.agent_ctxs[agent_id]
        expr_processor = agent_ctx.expr_processor[policy_id]

        # 由于expr_processor类是单例，因此只用调用一次即可
        expr_processor.gen_expr(
            extra_info["must_need_sample_info"],
            extra_info["network_sample_info"],
            self.from_actor_model_version,
            self.from_learner_model_version,
        )

    def get_current_models(self):
        """
        获取当前配置的policy里的Agent
        """
        if CONFIG.wrapper_type == KaiwuDRLDefine.WRAPPER_LOCAL:
            for policy, model_wrapper in self.policy_model_wrapper_maps.items():
                self.current_models.append(model_wrapper.get_model_object())

        elif CONFIG.wrapper_type == KaiwuDRLDefine.WRAPPER_REMOTE:
            # 支持多个agent操作, 此时是多个model_wrapper对象
            for policy_name, policy_conf in AppConf[CONFIG.app].policies.items():
                algo = policy_conf.algo
                model = AlgoConf[algo].aisrv_model(
                    agent_type=CONFIG.svr_name,
                    logger=self.logger,
                    monitor=self.monitor_proxy if CONFIG.use_prometheus else None,
                )
                model.framework_handler = self

                self.current_models.append(model)
        else:
            pass

    # predict函数, 单机单进程版本使用
    def predict_local(self, agent, predict_data):
        # 根据policy获取对应的model_wrapper
        model_wrapper = self.policy_model_wrapper_maps.get(CONFIG.policy_name)
        return model_wrapper.predict_local(predict_data, predict=True)

    # exploit函数, 单机单进程版本使用
    def exploit_local(self, agent, predict_data):
        # 根据policy获取对应的model_wrapper
        model_wrapper = self.policy_model_wrapper_maps.get(CONFIG.policy_name)
        return model_wrapper.exploit_local(predict_data, predict=True)

    # 保存模型的函数, 单机单进程版本使用
    def save_param(
        self,
        agent,
        func,
        *args,
        source=KaiwuDRLDefine.SAVE_OR_LOAD_MODEL_By_USER,
        **kargs,
    ):
        # 根据policy获取对应的model_wrapper
        model_wrapper = self.policy_model_wrapper_maps.get(CONFIG.policy_name)
        return model_wrapper.save_param(agent, func, *args, source=source, **kargs)

    # train 函数, 单机单进程版本使用
    def train_local(self, agent, data, *args, **kargs):
        # 根据policy获取对应的model_wrapper
        model_wrapper = self.policy_model_wrapper_maps.get(CONFIG.policy_name)
        model_wrapper.train_local(data, *args, **kargs)

        # 对于单机单进程来说下面的需要上报监控值
        if CONFIG.wrapper_type == KaiwuDRLDefine.WRAPPER_LOCAL:
            self.train_predict_stat()

    # 上报监控数据, 因为单机单进程情况下需要从aisrv上报监控数据, 集群版本则分aisrv, actor, learner上报
    def train_predict_stat(self):
        if int(CONFIG.use_prometheus):
            now = time.time()
            if now - self.last_report_monitor_time >= CONFIG.prometheus_stat_per_minutes * 60:
                predict_count = 0
                train_count = 0
                for policy, model_wrapper in self.policy_model_wrapper_maps.items():
                    predict_count += model_wrapper.predict_stat
                    train_count += model_wrapper.train_stat

                sample_production_and_consumption_ratio = 0
                if predict_count > 0:
                    sample_production_and_consumption_ratio = round(train_count / predict_count, 3)

                monitor_data = {
                    KaiwuDRLDefine.MONITOR_TRAIN_SUCCESS_CNT: train_count,
                    KaiwuDRLDefine.MONITOR_TRAIN_GLOBAL_STEP: train_count,
                    KaiwuDRLDefine.MONITOR_ACTOR_PREDICT_SUCC_CNT: predict_count,
                    KaiwuDRLDefine.SAMPLE_PRODUCTION_AND_CONSUMPTION_RATIO: sample_production_and_consumption_ratio,
                    KaiwuDRLDefine.MONITOR_SENDTO_REVERB_SUCC_CNT: train_count,
                }

                self.monitor_proxy.put_data({self.current_pid: monitor_data})
                self.last_report_monitor_time = now

    def standard_load_last_new_model(self, agent, func, *args, **kargs):
        """
        单机单进程版本的没有learner/actor之间的model文件传递, 但是需要在评估时加载model文件
        """
        model_wrapper = self.policy_model_wrapper_maps.get(CONFIG.policy_name)
        model_wrapper.standard_load_last_new_model(agent, func, *args, **kargs)

    # before_run函数
    def before_run(self):
        # 注意传入的参数格式
        kaiwu_env.setup(run_mode="proxy", skylarena_url=f"tcp://{self.simu_ctx.client_address}")

        # 如果是KaiwuDRLDefine.WRAPPER_LOCAL模式需要走下面逻辑
        if CONFIG.wrapper_type == KaiwuDRLDefine.WRAPPER_LOCAL:
            # policy_name 主要是和conf/app_conf.json设置一致
            self.policy_conf = AppConf[CONFIG.app].policies

            # 创建model_wrapper
            create_standard_model_wrapper(
                self.policy_conf,
                self.policy_model_wrapper_maps,
                None,
                self.logger,
                self.monitor_proxy,
            )

            # 清空id_list文件, 否则文件会持续增长
            clear_id_list_file(framework=True)

            model_wrapper = self.policy_model_wrapper_maps.get(CONFIG.policy_name)

            # 第一次保存模型时id的默认值即0
            model_wrapper.save_param_by_source(source=KaiwuDRLDefine.SAVE_OR_LOAD_MODEL_By_FRAMEWORK)

            # 更新id_list文件
            update_id_list(0, framework=True)

            # 清空使用者保存的文件目录
            clear_user_ckpt_dir()

            # eval下加载业务侧的模型文件
            if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODEL_EVAL:
                self.load_model_common_object = LoadModelCommon(self.logger)
                self.load_model_common_object.set_policy_model_wrapper_maps(self.policy_model_wrapper_maps)
                self.load_model_common_object.standard_load_last_new_model_by_framework_local()

        # 获取当前的model对象
        self.get_current_models()

        return True

    # 设置下run_time
    def init_agent_runtime(self, agents, envs):
        # 设置下agent阵营信息
        self.start_all_agents(agents)

    def workflow(self):
        """
        该函数主要是标准化调用的run函数, 由于是使用者调用的, 故这里需要加上处理Error, Warning的逻辑
        """
        error_code = KaiwuDRLDefine.DOCKER_EXIT_CODE_SUCCESS

        with warnings.catch_warnings():
            warnings.simplefilter("error", category=RuntimeWarning)

            try:
                # 记录运行时间
                process_start_time = time.monotonic()

                # 创建游戏环境
                env = kaiwu_env.make(CONFIG.app)

                # 创建智能体
                self.init_agent_runtime(self.current_models, [env])

                if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODEL_TRAIN:
                    # 直接调用训练的workflow
                    AlgoConf[CONFIG.algo].train_workflow([env], self.current_models, self.logger, self.monitor_proxy)

                else:
                    # 直接调用配置的评估的workflow
                    AlgoConf[CONFIG.algo].eval_workflow([env], self.current_models, self.logger)

                error_code = KaiwuDRLDefine.DOCKER_EXIT_CODE_SUCCESS

            except RuntimeError:
                self.logger.error(
                    f"kaiwu_rl_helper workflow() RuntimeError Exception, "
                    f"traceback.print_exc() is {traceback.format_exc()}"
                )

                error_code = KaiwuDRLDefine.DOCKER_EXIT_CODE_ERROR

            except Exception as e:
                self.logger.error(
                    f"kaiwu_rl_helper workflow() Exception {str(e)}, traceback.print_exc() is {traceback.format_exc()}"
                )

                error_code = KaiwuDRLDefine.DOCKER_EXIT_CODE_ERROR

            finally:

                # aisrv与learner进行通信, 告诉其需要安全退出
                self.send_process_stop_request(self.current_models, error_code)

                self.logger.info("kaiwu_rl_helper finally")

                time.sleep(CONFIG.handle_sigterm_sleep_seconds)

    # run 主函数
    def run(self):
        if not self.before_run():
            self.logger.error(f"kaiwu_rl_helper before_run failed, so return")
            return

        self.agent_ctxs = {}
        self.simu_ctx.agent_ctxs = self.agent_ctxs

        """
        调用业务侧的主函数代码
        """
        self.workflow()

    # predict_detail函数, 因为exploit和predict函数通用
    def predict_detail(self, agent, predict_data):
        if not predict_data:
            return None

        # 传递过来的predict_data先进行序列化再处理, 因为可能是不同的数据结构的
        predict_datas = {
            KaiwuDRLDefine.MESSAGE_TYPE: KaiwuDRLDefine.MESSAGE_PREDICT,
            KaiwuDRLDefine.MESSAGE_VALUE: dill.dumps(predict_data),
        }

        """
        针对单个agent的数据预测
        """
        agent_ctx = agent.ctx
        agent_id = agent_ctx.agent_id
        for policy_id in agent_ctx.policy:
            # 调用AsyncPolicy的send_pred_data函数
            success, actor_address = agent_ctx.policy[policy_id].send_pred_data(self.slot_id, predict_datas, agent_ctx)
            # self.logger.debug(f'kaiwu_rl_helper aisrv {self.slot_id} send to actor: {predict_datas}')
            if not success:
                self.logger.error(
                    f"kaiwu_rl_helper policy_id {policy_id} agent_id {agent_id} "
                    f"send_pred_data to actor {actor_address} failed"
                )
                continue

        agent_ctx.pred_output = {}
        for policy_id in agent_ctx.policy:
            pred_output = agent_ctx.policy[policy_id].get_pred_result(self.slot_id, agent_ctx)

            # self.logger.debug(f'kaiwu_rl_helper aisrv {self.slot_id} recv from actor: {pred_output}')
            if not pred_output:
                self.logger.error("kaiwu_rl_helper get_pred_result failed")
            else:
                agent_ctx.pred_output[policy_id] = pred_output

        # 提取数据
        preds = []
        for policy_id in agent_ctx.policy:
            pred = dill.loads(agent_ctx.pred_output[policy_id][agent_id]["pred"])
            preds.append(pred)
            self.from_actor_model_version = agent_ctx.pred_output[policy_id][agent_id]["model_version"]

        return preds[0]

    # exploit函数, 集群版本使用
    def exploit(self, agent, predict_data):
        if not predict_data:
            return None

        return self.predict_detail(agent, predict_data)

    # predict函数, 集群版本使用
    def predict(self, agent, predict_data):
        if not predict_data:
            return None

        return self.predict_detail(agent, predict_data)

    # 发送样本数据, 集群版本使用
    def send_train_data(self, agent, train_data, train_data_prioritized):
        if not train_data:
            return

        """
        采用下面规则:
        1. 如果是对战, 默认是主策略上发送样本数据, 此时只有1个learner
        2. 如果是非对战的, 在其各自的策略上发送样本数据
        """
        if CONFIG.self_play:
            agent_ctx = next(iter(self.agent_ctxs.values()))
        else:
            agent_ctx = agent.ctx

        policy = agent_ctx.policy[agent_ctx.main_id]

        train_data_detail = {
            "train_data": train_data,
            "train_data_prioritized": train_data_prioritized,
        }
        train_data = {
            KaiwuDRLDefine.MESSAGE_TYPE: KaiwuDRLDefine.MESSAGE_TRAIN,
            KaiwuDRLDefine.MESSAGE_VALUE: train_data_detail,
        }

        policy.send_train_data(train_data, agent_ctx)

    # 发送保存模型文件到learner
    def send_save_model_file_data(self, agent, path=None, id=None):

        """
        采用下面规则:
        1. 如果是对战, 默认是主策略上发送样本数据, 此时只有1个learner
        2. 如果是非对战的, 在其各自的策略上发送样本数据
        """
        if CONFIG.self_play:
            agent_ctx = next(iter(self.agent_ctxs.values()))
        else:
            agent_ctx = agent.ctx

        policy = agent_ctx.policy[agent_ctx.main_id]
        save_model_data_detail = {
            "ip": f"{self.ip}_{self.slot_id}",
            "path": path,
            "id": id,
        }
        save_model_data = {
            KaiwuDRLDefine.MESSAGE_TYPE: KaiwuDRLDefine.MESSAGE_SAVE_MODEL,
            KaiwuDRLDefine.MESSAGE_VALUE: save_model_data_detail,
        }

        policy.send_train_data(save_model_data, agent_ctx)

    # 发送process_stop请求, 单机版本和集群版本都会使用到
    def send_process_stop_request(self, agents, error_code):
        """
        agents, 智能体集合
        error_code, 退出码, 分为正确的和错误的退出码
        """

        process_stop_request_detail = {"ip": f"{self.ip}_{self.slot_id}", "error_code": error_code}

        process_stop_request_data = {
            KaiwuDRLDefine.MESSAGE_TYPE: KaiwuDRLDefine.MESSAGE_PROCESS_STOP,
            KaiwuDRLDefine.MESSAGE_VALUE: process_stop_request_detail,
        }

        """
        采用下面规则:
        1. 如果是对战, 默认是主策略上发送样本数据, 此时只有1个learner
        2. 如果是非对战的, 在其各自的策略上发送样本数据
        """
        if CONFIG.self_play:
            agent_ctx = next(iter(self.agent_ctxs.values()))
        else:
            agent_ctx = agents[0].ctx
        policy = agent_ctx.policy[agent_ctx.main_id]

        policy.send_train_data(process_stop_request_data, agent_ctx)

    # 发送load_model请求, 集群版本使用
    def send_load_model_file_data(self, agent, path=None, id=None):

        # id必须为正常的
        if not check_id_valid(id):
            self.logger.error(f"kaiwu_rl_helper send_load_model_file_data failed, id {id} is not valid, please check")
            return

        # 无论是对战或者是非对战场景下预测进程都需要load_model
        agent_ctx = agent.ctx

        load_model_data_detail = {
            "ip": f"{self.ip}_{self.slot_id}",
            "policy": agent_ctx.main_id,
            "path": path,
            "id": id,
        }
        load_model_data = {
            KaiwuDRLDefine.MESSAGE_TYPE: KaiwuDRLDefine.MESSAGE_LOAD_MODEL,
            KaiwuDRLDefine.MESSAGE_VALUE: load_model_data_detail,
        }

        policy = agent_ctx.policy[agent_ctx.main_id]

        policy.send_pred_data(self.slot_id, load_model_data, agent_ctx)

    def stop(self):
        self.exit_flag.value = True
        for __, policy in self.policies.items():
            policy.stop()

        # 上报监控指标
        monitor_data = {}
        for key, value in self.app_monitor_data.items():
            monitor_data[key] = value

        if monitor_data:
            self.monitor_proxy.put_data(monitor_data)

        time.sleep(1)

        self.logger.info("kaiwu_rl_helper success stop")

    def normalize_policy_ids(self, ids):
        assert isinstance(ids, (str, list)), "only str or list of str is supported"
        if isinstance(ids, str):
            ids = [ids]
        return ids

    def stop_all_agents(self, agents):
        """
        停止所有的agents
        """
        for i, agent in enumerate(agents):
            self.stop_agent(i)

    # 单个agent_id的停止
    def stop_agent(self, agent_id):
        self.logger.info(f"kaiwu_rl_helper stop agent {agent_id}")
        agent_ctx = self.agent_ctxs[agent_id]

        policy_ids = self.normalize_policy_ids(self.env.policy_mapping_fn(agent_id))
        for policy_id in policy_ids:
            if agent_ctx.policy[policy_id].need_train():
                if not self.use_sample_server:
                    agent_ctx.expr_processor[policy_id].finalize()

        del self.agent_ctxs[agent_id]

    def start_all_agents(self, agents):
        """
        启动所有的agents, 主要是对齐agent和policy的关系
        """
        for i, agent in enumerate(agents):
            # 这里强制做了下赋值即framework_handler设置为self
            agent.framework_handler = self

            self.start_agent(i)
            agent.ctx = self.agent_ctxs[i]

    # 单个agent_id的启动
    def start_agent(self, agent_id):
        """
        形如以下配置
        "policies": {
                    "train": {
                        "policy_builder": "kaiwudrl.server.aisrv.async_policy.AsyncBuilder",
                        "algo": "ppo",
                        "state": "app.gym.gym_proto.GymState",
                        "action": "app.gym.gym_proto.GymAction",
                        "reward": "app.gym.gym_proto.GymReward",
                        "actor_network": "app.gym.gym_network.GymDeepNetwork",
                        "learner_network": "app.gym.gym_network.GymDeepNetwork",
                        "reward_shaper": "app.gym.gym_reward_shaper.GymRewardShaper",
                        "eigent_value": "app.gym.gym_eigent_value.GymEigentValue"
                    },
                    "predict": {
                        "policy_builder": "kaiwudrl.server.aisrv.async_policy.AsyncBuilder",
                        "algo": "ppo",
                        "state": "app.gym.gym_proto.GymState",
                        "action": "app.gym.gym_proto.GymAction",
                        "reward": "app.gym.gym_proto.GymReward",
                        "actor_network": "app.gym.gym_network.GymDeepNetwork",
                        "learner_network": "app.gym.gym_network.GymDeepNetwork",
                        "reward_shaper": "app.gym.gym_reward_shaper.GymRewardShaper",
                        "eigent_value": "app.gym.gym_eigent_value.GymEigentValue"
                    }

        """
        agent_ctx = AgentContext()
        agent_ctx.done = False
        agent_ctx.agent_id = agent_id

        # 设置主要main_id, policy_ids为策略列表
        np.random.seed(int(time.time() * 1000) % (2**20))
        policy_ids = list(AppConf[CONFIG.app].policies.keys())

        # 每一个agent在启动时就需要确定唯一的policy，但是两边对弈的agent可以是不同policy
        if int(CONFIG.self_play):
            # 如果agent为self_play_agent那么其策略为策略列表中的对应策略
            if agent_id == CONFIG.self_play_agent_index:
                agent_ctx.main_id = policy_ids[CONFIG.self_play_agent_index]
                policy_ids = [policy_ids[CONFIG.self_play_agent_index]]
                assert agent_ctx.main_id == CONFIG.self_play_policy, "Check your config of self_play_policy"

            elif agent_id == CONFIG.self_play_old_agent_index:
                # 当agent_id为对手策略时，80%设置为新策略，20%为旧策略
                if np.random.uniform() <= (1 - float(CONFIG.self_play_new_ratio)):
                    agent_ctx.main_id = policy_ids[CONFIG.self_play_old_agent_index]
                    policy_ids = [policy_ids[CONFIG.self_play_old_agent_index]]
                    assert agent_ctx.main_id == CONFIG.self_play_old_policy, "Check your config of self_play_old_policy"
                else:
                    agent_ctx.main_id = policy_ids[CONFIG.self_play_agent_index]
                    policy_ids = [policy_ids[CONFIG.self_play_agent_index]]
                    assert agent_ctx.main_id == CONFIG.self_play_policy, "Check your config of self_play_policy"
        else:
            # 如果不是self-play模式，那么agent自动加载第一种policy
            agent_ctx.main_id = policy_ids[0]
            policy_ids = [policy_ids[0]]

        self.logger.info(f"kaiwu_rl_helper start agent {agent_id} with {policy_ids[0]}")

        """ policy conf, 形如
                    "train_one": {
                        "policy_builder": "kaiwudrl.server.aisrv.async_policy.AsyncBuilder",
                        "algo": "ppo",
                        "state": "app.gym.gym_proto.GymState",
                        "action": "app.gym.gym_proto.GymAction",
                        "reward": "app.gym.gym_proto.GymReward",
                        "actor_network": "app.gym.gym_network.GymDeepNetwork",
                        "learner_network": "app.gym.gym_network.GymDeepNetwork",
                        "reward_shaper": "app.gym.gym_reward_shaper.GymRewardShaper",
                        "eigent_value": "app.gym.gym_eigent_value.GymEigentValue"
                    }
        """
        agent_ctx.policy_conf = {}
        """
        policy, 形如"policy_builder": "kaiwudrl.server.aisrv.async_policy.AsyncBuilder",
        """
        agent_ctx.policy = {}
        # 预测的响应结果
        agent_ctx.pred_output = {}

        agent_ctx.start_time = time.monotonic()

        # aisrv发送给actor的message id, 从1自增
        agent_ctx.message_id = 1

        # aisrv发送给actor的model_version, 由actor负责赋值
        agent_ctx.model_version = -1

        # policy_ids的列表长度根据运行模式不一致, 比如self-play是1, 非self-play的需要看具体情况
        for policy_id in policy_ids:
            policy_conf = AppConf[CONFIG.app].policies[policy_id]
            policy = self.policies[policy_id]
            agent_ctx.policy_conf[policy_id] = policy_conf
            agent_ctx.policy[policy_id] = policy

            if policy.need_train():
                assert hasattr(policy_conf, "algo"), "trainable policy need to specify algo"
                """
                 {
                     "ppo": {
                         "actor_model": "kaiwudrl.common.algorithms.model.Model",
                         "learner_model": "kaiwudrl.common.algorithms.model.Model",
                         "trainer": "kaiwudrl.server.learner.ppo_trainer.PPOTrainer",
                         "predictor": "kaiwudrl.server.actor.ppo_predictor.PPOPredictor",
                         "expr_processor": "kaiwudrl.common.algorithms.ppo_processor.PPOProcessor",
                         "default_config": "kaiwudrl.common.algorithms.ppo.PPODefaultConfig"
                     }
                 }
                 """

        self.agent_ctxs[agent_id] = agent_ctx

    def handle_sigterm(self, sig, frame):
        self.stop()
        model_wrapper = self.policy_model_wrapper_maps.get(CONFIG.policy_name)
        self.logger.info(f"kaiwu_rl_helper {os.getpid()} is starting to handle the SIGTERM signal.")
        model_wrapper.save_param_by_source(source=KaiwuDRLDefine.SAVE_OR_LOAD_MODEL_BY_SIGTERM)
        # 处理完保存最新模型,等待其他进程工作,避免pod提前退出
        time.sleep(CONFIG.handle_sigterm_sleep_seconds)
