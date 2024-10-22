#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file actor_server_sync.py
# @brief
# @author kaiwu
# @date 2023-11-28


"""
gevent代码运行时影响比较大, 慎用
import gevent
from gevent import monkey
"""

import os
import asyncio
import random
import threading
import multiprocessing
import datetime
import traceback
import schedule
from kaiwudrl.common.ipc.zmq_util import ZmqServer, ZmqPoller
from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.utils.slots import Slots
from kaiwudrl.common.pybind11.zmq_ops import *
from kaiwudrl.common.utils.common_func import (
    TimeIt,
    Context,
    set_schedule_event,
)
from kaiwudrl.common.logging.kaiwu_logger import KaiwuLogger, g_not_server_label
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine


class ActorServerSync(multiprocessing.Process):
    """
    该类主要用于actor <--> aisrv之间的消息处理:
    1. aisrv <-- actor方向, 采用self.zmq_client
    2. aisrv --> actor方向
    2.1 如果是框架定义session, 采用self.zmq_ops_client
    2.2 如果是业务定义session, 采用self.zmq_client
    """

    def __init__(self) -> None:
        super(ActorServerSync, self).__init__()

        # actor建立zmq, 从aisrv收到请求, 并且处理后返回给aisrv, 注意其端口必须和aisrv启动的zmq端口一致
        self.zmq_server = ZmqServer(CONFIG.ip_address, CONFIG.zmq_server_port)

        # 单个zmq server处理多个zmq client的请求和响应
        self.client_id = ""

        # 停止标志位
        self.exit_flag = multiprocessing.Value("b", False)

        # 上下文Context
        self.context = Context()
        self.context.slots = Slots(CONFIG.max_tcp_count)
        self.context.slot_group_name = "actor_server"
        self.context.slots.register_group(self.context.slot_group_name)

        """
        multiprocessing.Manager()类型的建议少用, 可能会出现多进程之间的问题
        """
        # self.manager = multiprocessing.Manager()

        # 使用队列模式, 获取actor的预测响应, 发给aisrv
        self.predict_result_queue = multiprocessing.Queue(CONFIG.queue_size)
        if CONFIG.server_use_processes == KaiwuDRLDefine.RUN_AS_THREAD:
            self.predict_result_condition = multiprocessing.Condition()
            self.lock = threading.Lock()

        """
        if int(CONFIG.use_pipeline_predict):
            self.predict_result_queue = multiprocessing.Queue(CONFIG.queue_size)

        else:
            self.input_pipe, self.output_pipe = multiprocessing.Pipe(duplex=False)
        """

        # 采用队列模式, actor接收来自aisrv的zmq请求
        if CONFIG.aisrv_actor_communication_way == KaiwuDRLDefine.COMMUNICATION_WAY_ZMQ:
            if CONFIG.actor_server_predict_server_different_queue:
                self.predict_request_queues = None
            else:
                self.predict_request_queue = multiprocessing.Queue(CONFIG.queue_size)

        # 统计指标
        self.send_to_aisrv_succ_cnt = 0
        self.send_to_aisrv_error_cnt = 0

        self.recv_from_aisrv_succ_cnt = 0
        self.recv_from_aisrv_error_cnt = 0

        # actor --> aisrv发送时批量处理耗时
        self.actor_send_to_aisrv_batch_cost_time_ms = 0

        # 采用压缩算法时, 压缩耗时, 解压缩耗时, 压缩大小
        self.max_compress_time = 0
        self.max_decompress_time = 0
        self.max_compress_size = 0

        # actor_server从zmq获取数据放入本地队列报错次数
        self.actor_server_queue_full_cnt = 0

        # 如果是队列设置在predict里, 则actor_server需要记录下当前的predict_queue的
        if CONFIG.actor_server_predict_server_different_queue:
            self.predict_request_queue = None

        # 进程pid
        self.current_pid = 0

    def set_predict_pipe_conns(self, predict_pipe_conns):
        self.predict_pipe_conns = predict_pipe_conns

    def set_predict_request_queues(self, predict_request_queues):
        self.predict_request_queues = predict_request_queues

    # actor --> aisrv发送预测请求的响应
    def put_predict_result_data(self, predict_result_data):
        if predict_result_data:
            # 采用pickle序列化, 因为采用pipe可以去掉序列化和反序列化步骤
            # predict_result_data = pickle.dumps(predict_result_data)

            if self.predict_result_queue.full():
                # 队列满时报错
                return
            else:
                self.predict_result_queue.put(predict_result_data)

    # 返回类的名字, 便于确认调用关系
    def get_class_name(self):
        return self.__class__.__name__

    def before_run(self):

        # 日志处理
        self.logger = KaiwuLogger()
        self.current_pid = os.getpid()
        self.logger.setLoggerFormat(
            f"/{CONFIG.svr_name}/actor_server_pid{self.current_pid}_log_"
            f"{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log",
            CONFIG.svr_name,
        )

        # 注意调用时序关系
        self.zmq_server.bind()
        self.logger.info(
            f"actor_server zmq server bind at {CONFIG.ip_address} : {CONFIG.zmq_server_port}",
            g_not_server_label,
        )

        # zmq的poller使用
        self.zmq_poller = ZmqPoller()
        self.zmq_server_socket = self.zmq_server._socket
        self.zmq_poller.register_read(self.zmq_server_socket)

        # 启动记录发送成功失败的数目的定时器
        self.send_and_recv_zmq_stat()

        # 进程空转了N次就主动让出CPU, 避免CPU空转100%
        self.process_run_idle_count = 0

        """
        # 启动类内线程, 处理预测请求/响应收发, 但是效果不明显
        self.actor_receiver = threading.Thread(target=ActorServerSync.actor_receive_msg, args=(self,))
        self.actor_receiver.start()
        self.actor_sender = threading.Thread(target=ActorServerSync.actor_send_msg, args=(self,))
        self.actor_sender.start()
        """

        # 在before run最后打印启动成功日志
        self.logger.info(
            (
                f"actor_server is start success at pid is {self.current_pid}, "
                f"class name is {self.get_class_name()}, "
                f"use process type is {CONFIG.server_use_processes}"
            ),
            g_not_server_label,
        )

        return True

    def set_monitor_proxy(self, monitor_proxy):
        self.monitor_proxy = monitor_proxy

    # 定时器采用schedule, need pip install schedule
    def send_and_recv_zmq_stat(self):

        set_schedule_event(CONFIG.prometheus_stat_per_minutes, self.zmq_stat)

    def zmq_stat_reset(self):

        self.send_to_aisrv_succ_cnt = 0
        self.send_to_aisrv_error_cnt = 0
        self.recv_from_aisrv_succ_cnt = 0
        self.recv_from_aisrv_error_cnt = 0
        self.actor_server_queue_full_cnt = 0

        self.actor_send_to_aisrv_batch_cost_time_ms = 0
        self.max_compress_time = 0
        self.max_decompress_time = 0
        self.max_compress_size = 0

    def zmq_stat(self):

        # 针对zmq_server的统计
        if int(CONFIG.use_prometheus):
            actor_server_request_queue_size = 0
            actor_server_result_queue_size = 0
            try:
                actor_server_request_queue_size = self.predict_request_queue.qsize()
                actor_server_result_queue_size = self.predict_result_queue.qsize()
            except Exception as e:
                pass

            actor_send_to_aisrv_batch_cost_time_ms = self.actor_send_to_aisrv_batch_cost_time_ms
            monitor_data = {
                KaiwuDRLDefine.MONITOR_ACTOR_SENDTO_AISRV_SUCC_CNT: self.send_to_aisrv_succ_cnt,
                KaiwuDRLDefine.MONITOR_ACTOR_SENDTO_AISRV_ERROR_CNT: self.send_to_aisrv_error_cnt,
                KaiwuDRLDefine.MONITOR_ACTOR_RECEIVEFROM_AISRV_SUCC_CNT: self.recv_from_aisrv_succ_cnt,
                KaiwuDRLDefine.MONITOR_ACTOR_RECEIVEFROM_AISRV_ERROR_CNT: self.recv_from_aisrv_error_cnt,
                KaiwuDRLDefine.MONITOR_ACTOR_SENDTO_AISRV_BATCH_COST_TIME_MS: actor_send_to_aisrv_batch_cost_time_ms,
                # KaiwuDRLDefine.MONITOR_ACTOR_MAX_COMPRESS_TIME : self.max_compress_time,
                # KaiwuDRLDefine.MONITOR_ACTOR_MAX_DECOMPRESS_TIME : self.max_decompress_time,
                # KaiwuDRLDefine.MONITOR_ACTOR_MAX_COMPRESS_SIZE : self.max_compress_size,
                KaiwuDRLDefine.MONITOR_ACTOR_SERVER_QUEUE_FULL_CNT: self.actor_server_queue_full_cnt,
                KaiwuDRLDefine.MONITOR_ACTOR_SERVER_REQUEST_QUEUE_SIZE: actor_server_request_queue_size,
                KaiwuDRLDefine.MONITOR_ACTOR_SERVER_RESULT_QUEUE_SIZE: actor_server_result_queue_size,
            }
            self.monitor_proxy.put_data({self.current_pid: monitor_data})

        """
        self.logger.debug(f'actor_server zmq stat, send_succ_cnt is {self.send_to_aisrv_succ_cnt}, \
                          send_error_cnt is {self.send_to_aisrv_error_cnt} \
                          recv_succ cnt is {self.recv_from_aisrv_succ_cnt} \
                          recv_error_cnt is {self.recv_from_aisrv_error_cnt}', g_not_server_label)
        """

        # 指标复原, 计算的是周期性的上报指标
        self.zmq_stat_reset()

    def get_from_to_predict_queue(self):
        """
        这里的分2种情况:
        1. 如果是on-polciy的, 注意需要采用非阻塞的, 否则阻塞了predict主流程
        2. 如果是非on-polciy的, 注意采用阻塞的, 性能要好于非阻塞的, 但是对于battlesrv数量远远小于了batch_size则需要设置超时时间
        """
        if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
            try:
                return self.predict_request_queue.get_nowait()
            except Exception as e:
                return None

        else:
            try:
                return self.predict_request_queue.get(timeout=CONFIG.queue_timeout)
            except Exception as e:
                return None

    # 处理aisrv --> actor方向预测请求, 线程模式
    def actor_receive_msg_thread(self):
        while True:
            socks = dict(self.zmq_poller.get_poller().poll(timeout=0))
            if (
                self.zmq_server_socket in socks
                and socks[self.zmq_server_socket] == self.zmq_poller.get_zmq_pollin_state()
            ):
                self.actor_receive_msg_direct()

    def put_data_to_queue(self, data):
        """
        放入队列, 分下面场景:
        1. 如果queue在actor_server本地, 则放入actor_server本地
        2. 如果queue在predict里, 则放入predict
        """
        if not data:
            return

        if CONFIG.actor_server_predict_server_different_queue:
            random_number = int(random.uniform(0, CONFIG.actor_predict_process_num))
            self.predict_request_queues[random_number].put(data)
        else:
            self.predict_request_queue.put(data)

    def actor_receive_msg_direct_by_zmq_poller(self):
        """
        参照zmq poller高级用法的

        poll() 方法的主要优势在于它允许你同时监视多个套接字，并在有可读性变化时进行处理。这对于同时处理多个套接字的情况非常有用，或者当你需要在等待期间执行其他任务时。
        然而，如果你只需要处理一个套接字，并且没有其他需要处理的任务，那么直接调用 recv() 函数可能更简单和高效。故高性能场景不建议调用该函数
        """
        socks = dict(self.zmq_poller.get_poller().poll(timeout=0))
        if self.zmq_server_socket in socks and socks[self.zmq_server_socket] == self.zmq_poller.get_zmq_pollin_state():
            self.actor_receive_msg_direct()

    # 处理aisrv --> actor方向预测请求, 直接调用
    def actor_receive_msg_direct(self):
        """
        actor --> aisrv返回响应, 因为只是需要获取client_id, 故不需要处理发送来的msg
        因为不能阻塞流程, 故设置block为False, 本次出现读取异常, 则下一个循环接着处理
        """
        try:
            if CONFIG.server_use_processes == KaiwuDRLDefine.RUN_AS_THREAD:
                self.lock.acquire()

            self.client_id, message = self.zmq_server.recv(block=False, binary=True)

            if CONFIG.server_use_processes == KaiwuDRLDefine.RUN_AS_THREAD:
                self.lock.release()

            """
            if self.client_id:
                self.logger.debug(f'actor_server zmq server recv a new msg from {self.client_id} success', g_not_server_label)
            """

            if message and CONFIG.aisrv_actor_communication_way == KaiwuDRLDefine.COMMUNICATION_WAY_ZMQ:
                self.recv_from_aisrv_succ_cnt += 1

                if CONFIG.distributed_tracing:
                    self.logger.info(
                        f"actor_server distributed_tracing actor_receive_msg_direct recv a new message start",
                        g_not_server_label,
                    )

                try:
                    # 直接放入原始的数据, 在on_policy_predictor进程里解压缩和压缩, 减少CPU损耗
                    self.put_data_to_queue(message)
                except Exception as e:
                    # 当actor_server --> predict的队列满时报错
                    self.logger.error(
                        f"actor_server zmq server to predict queue error, message is {str(e)}",
                        g_not_server_label,
                    )
                    self.actor_server_queue_full_cnt += 1

                if CONFIG.distributed_tracing:
                    self.logger.info(
                        f"actor_server distributed_tracing actor_receive_msg_direct recv a new message end",
                        g_not_server_label,
                    )

            # 未来扩展
            else:
                pass

        except Exception as e:
            # 这里暂时没有请求aisrv请求是正常现象, 下一个循环接着处理
            self.process_run_idle_count += 1
            pass

    # 处理aisrv --> actor方向预测请求, 协程模式
    async def actor_receive_msg_by_coroutine(self):
        """
        actor --> aisrv返回响应, 因为只是需要获取client_id, 故不需要处理发送来的msg
        因为不能阻塞流程, 故设置block为False, 本次出现读取异常, 则下一个循环接着处理
        """
        try:
            self.client_id, message = self.zmq_server.recv(block=False, binary=True)

            """
            if self.client_id:
                self.logger.debug(
                    f'actor_server zmq server recv a new msg from {self.client_id} '
                    f'success', g_not_server_label
                )
            """

            if message and CONFIG.aisrv_actor_communication_way == KaiwuDRLDefine.COMMUNICATION_WAY_ZMQ:
                self.recv_from_aisrv_succ_cnt += 1

                if CONFIG.distributed_tracing:
                    self.logger.info(
                        f"actor_server distributed_tracing actor_receive_msg_direct a new message",
                        g_not_server_label,
                    )

                try:
                    # 直接放入原始的数据, 在on_policy_predictor进程里解压缩和压缩, 减少CPU损耗
                    await self.predict_request_queue.put(message)
                except Exception as e:
                    # 当actor_server --> predict的队列满时报错
                    self.logger.error(
                        f"actor_server zmq server to predict queue is full",
                        g_not_server_label,
                    )
                    self.actor_server_queue_full_cnt += 1

            # 未来扩展
            else:
                pass

        except Exception as e:
            # 这里暂时没有请求aisrv请求是正常现象, 下一个循环接着处理
            self.process_run_idle_count += 1
            pass

    # actor给aisrv发送预测结果函数, 协程模式
    async def actor_send_msg_to_aisrv_detail_by_coroutine(self, data):
        if not data:
            self.process_run_idle_count += 1
            return

        client_id, compressed_data = data
        with TimeIt() as ti:
            if CONFIG.server_use_processes == KaiwuDRLDefine.RUN_AS_THREAD:
                self.lock.acquire()

            self.zmq_server.send(str(client_id), compressed_data, binary=True)

            if CONFIG.server_use_processes == KaiwuDRLDefine.RUN_AS_THREAD:
                self.lock.release()

        # 获取采集周期里的最大值
        if self.actor_send_to_aisrv_batch_cost_time_ms < ti.interval * 1000:
            self.actor_send_to_aisrv_batch_cost_time_ms = ti.interval * 1000

        self.send_to_aisrv_succ_cnt += 1

        if CONFIG.distributed_tracing:
            self.logger.info(
                f"actor_server distributed_tracing zmq server send a new msg success",
                g_not_server_label,
            )

    # actor给aisrv发送预测结果函数, 分为流水线和非流水线的场景
    def actor_send_msg_to_aisrv_detail(self, data):
        if not data:
            self.process_run_idle_count += 1
            return

        client_id, compressed_data = data
        with TimeIt() as ti:
            if CONFIG.server_use_processes == KaiwuDRLDefine.RUN_AS_THREAD:
                self.lock.acquire()

            self.zmq_server.send(str(client_id), compressed_data, binary=True)

            if CONFIG.server_use_processes == KaiwuDRLDefine.RUN_AS_THREAD:
                self.lock.release()

        # 获取采集周期里的最大值
        if self.actor_send_to_aisrv_batch_cost_time_ms < ti.interval * 1000:
            self.actor_send_to_aisrv_batch_cost_time_ms = ti.interval * 1000

        self.send_to_aisrv_succ_cnt += 1

        if CONFIG.distributed_tracing:
            self.logger.info(
                f"actor_server distributed_tracing zmq server send a new msg success",
                g_not_server_label,
            )

    # 处理actor --> aisrv方向预测请求, 线程模式
    def actor_send_msg_thread(self):
        while True:
            with self.predict_result_condition:
                self.predict_result_condition.wait()

            # 一直循环直到队列为空
            while not self.predict_result_queue.empty():
                self.actor_send_msg_to_aisrv_detail(self.predict_result_queue.get())

    # 处理actor --> aisrv方向预测请求, 直接调用
    def actor_send_msg_direct(self):

        # 判断队列为空self.msg_queue.empty()时, 可能出现报错Connection reset by peer, 需要使用try-except形式
        try:
            data = self.predict_result_queue.get_nowait()
        except Exception as e:
            self.process_run_idle_count += 1
            return

        self.actor_send_msg_to_aisrv_detail(data)

    # 处理actor --> aisrv方向预测请求, 协程形式
    async def actor_send_msg_by_coroutine(self):
        # 判断队列为空self.msg_queue.empty()时, 可能出现报错Connection reset by peer, 需要使用try-except形式
        try:
            data = self.predict_result_queue.get_nowait()
        except Exception as e:
            self.process_run_idle_count += 1
            return

        await self.actor_send_msg_to_aisrv_detail_by_coroutine(data)

    def run_once_direct(self):

        # 进行预测请求/响应收发
        self.actor_receive_msg_direct()

        self.actor_send_msg_direct()

        # 记录发送给aisrv成功失败数目, 包括发出去和收回来的请求
        schedule.run_pending()

    """
    gevent代码运行时影响比较大, 慎用
    def run_once_gevent(self):

        # 进行预测请求/响应收发
        greenlet_send = gevent.spawn(self.actor_send_msg_direct)

        greenlet_receive = gevent.spawn(self.actor_receive_msg_direct)

        gevent.joinall([greenlet_send, greenlet_receive])

        # 记录发送给aisrv成功失败数目, 包括发出去和收回来的请求
        schedule.run_pending()
    """

    async def run_once_by_coroutine(self):

        # 进行预测请求/响应收发
        await asyncio.gather(self.actor_receive_msg_by_coroutine(), self.actor_send_msg_by_coroutine())

        # 记录发送给aisrv成功失败数目, 包括发出去和收回来的请求
        schedule.run_pending()

    def run_once_thread(self):

        # 记录发送给aisrv成功失败数目, 包括发出去和收回来的请求
        schedule.run_pending()

    def run(self):
        if not self.before_run():
            self.logger.error(f"actor_server before_run failed, so return", g_not_server_label)
            return

        # 线程模式下执行
        if CONFIG.server_use_processes == KaiwuDRLDefine.RUN_AS_THREAD:
            self.actor_receiver = threading.Thread(target=ActorServerSync.actor_receive_msg_thread, args=(self,))
            self.actor_receiver.daemon = True
            self.actor_receiver.start()
            self.actor_sender = threading.Thread(target=ActorServerSync.actor_send_msg_thread, args=(self,))
            self.actor_sender.daemon = True
            self.actor_sender.start()

        if CONFIG.server_use_processes == KaiwuDRLDefine.RUN_AS_COROUTINE:
            loop = asyncio.get_event_loop()

        while not self.exit_flag.value:
            try:
                if CONFIG.server_use_processes == KaiwuDRLDefine.RUN_AS_COROUTINE:
                    loop.run_until_complete(self.run_once_by_coroutine())
                elif CONFIG.server_use_processes == KaiwuDRLDefine.RUN_AS_DIRECT:
                    self.run_once_direct()
                elif CONFIG.server_use_processes == KaiwuDRLDefine.RUN_AS_THREAD:
                    self.run_once_thread()
                elif CONFIG.server_use_processes == KaiwuDRLDefine.RUN_AS_GEVENT:
                    pass
                    # self.run_once_gevent()

                # 未来扩展
                else:
                    pass

                """
                # 短暂sleep, 规避容器里进程CPU使用率100%问题, 由于actor的zmq_server是比较忙碌的, 这里暂时不做人为休眠, 后期修改为事件提醒机制
                if self.process_run_idle_count % CONFIG.idle_sleep_count == 0:
                    time.sleep(CONFIG.idle_sleep_second)

                    # process_run_count置0, 规避溢出
                    self.process_run_idle_count = 0
                """

            except Exception as e:
                self.logger.error(
                    f"actor_server ActorServer run error: {str(e)}, traceback.print_exc() is {traceback.format_exc()}",
                    g_not_server_label,
                )

    # 停止进程
    def stop(self):
        self.exit_flag.value = True
