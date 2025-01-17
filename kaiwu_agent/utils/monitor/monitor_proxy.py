#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import multiprocessing
import queue
import threading
import time
import os
import signal
import traceback
from kaiwu_agent.conf import ini_monitor as CONFIG
from kaiwu_agent.utils.monitor.prometheus_utils import PrometheusUtils

'''
此类用于进程与监控产品(当前是普罗米修斯, 后期可以按照需要调整)
 独立出进程, 减少核心路径消耗
'''


class MonitorProxy(threading.Thread):
    def __init__(self, logger) -> None:
        super(MonitorProxy, self).__init__()

        # 进程是否退出, 用于在异常条件下主动退出进程
        self.exit_flag = multiprocessing.Value('b', False)

        # 目前由于单个进程内有多个进程可能用到监控统计信息上报, 故日志需要区分下
        self.logger = logger
        self.logger.info(f'ppid is {threading.currentThread().ident}')

        '''
        该队列使用场景:
        1. aisrv,actor,learner主进程向该队列放置监控的数据
        2. monitor_proxy从该队列拿出需要监控的数据, 发往普罗米修斯
        '''
        self.msg_queue = queue.Queue(CONFIG.main.queue_size)

        self.walk = None

        # 按照时间间隔判断进程是否存活
        self.last_detection_time = time.time()

    def before_run(self):

        # PrometheusUtils 工具类, 与普罗米修斯交互操作
        self.prometheus_utils = PrometheusUtils(self.logger)

        self.process_run_count = 0

    '''
    monitor_data采用map形式, 即key/value格式, 监控指标/监控值
    '''

    def put_data(self, monitor_data):
        if not monitor_data:
            self.logger.error(f'monitor_data is None')
            return

        if self.msg_queue.full():
            self.logger.error(f'queue is full, return')
            return
        else:
            self.msg_queue.put(monitor_data)

    def get_data(self):
        if not self.msg_queue.empty():
            return self.msg_queue.get()

        return None

    def send_to_prometheus(self, monitor_data):
        if not monitor_data:
            return

        if not isinstance(monitor_data, dict):
            self.logger.error(f'monitor_data is not dict, return')
            return

        for monitor_name, montor_value in monitor_data.items():
            if isinstance(montor_value, list):
                for i in range(len(montor_value)):
                    self.prometheus_utils.gauge_use(
                        CONFIG.main.svr_name, monitor_name, monitor_name, montor_value[i])
            else:
                self.prometheus_utils.gauge_use(
                    CONFIG.main.svr_name, monitor_name, monitor_name, montor_value)

        self.prometheus_utils.push_to_prometheus_gateway()
        self.logger.debug(f'monitor_proxy push_to_prometheus_gateway success')

    def run_once(self):

        # 获取需要监控的数据
        monitor_data = self.get_data()
        if monitor_data:
            self.send_to_prometheus(monitor_data)

        # 进程alive检测, 包括aisrv和gamecore收包时间间隔
        if self.walk:
            stop = False
            now = time.time()

            if now - self.last_detection_time >= CONFIG.main.battlesrv_recv_response_from_gamecore_aisrv_seconds:
                if now - self.walk.last_recv_gamecore_time > CONFIG.main.battlesrv_recv_response_from_gamecore_aisrv_seconds:
                    self.logger.error(
                        f"Battlesrv pid {self.walk_battlesrv_pid} and alloc pid {self.walk_alloc_id} will killed because of gamecore response timeout")
                    stop = True
                
                if now - self.walk.last_recv_aisrv_time > CONFIG.main.battlesrv_recv_response_from_gamecore_aisrv_seconds:
                    self.logger.error(
                        f"Battlesrv pid {self.walk_battlesrv_pid} and alloc pid {self.walk_alloc_id} will killed because of aisrv response timeout")
                    stop = True
            
                if stop:        
                    # 安全停止gamecore
                    self.walk.controller.stop_game()

                    # 需要先停止alloc进程, 再停止battlesrv进程
                    if self.walk_alloc_id  > 0:
                        os.kill(self.walk_alloc_id, signal.SIGKILL)
                
                    if self.walk_battlesrv_pid > 0:
                        os.kill(self.walk_battlesrv_pid, signal.SIGKILL)
                
                self.last_detection_time = now

    def set_walk(self, walk, battlesrv_pid, alloc_pid):
        self.walk = walk
        self.walk_battlesrv_pid = battlesrv_pid
        self.walk_alloc_id = alloc_pid

    '''
    进程停止函数
    '''

    def stop(self):
        self.exit_flag.value = True
        self.join()

        self.logger.info('monitor_proxy MonitorProxy stop success')

    def run(self) -> None:
        self.before_run()

        while not self.exit_flag.value:
            try:
                self.run_once()

                # 短暂sleep, 规避容器里进程CPU使用率100%问题
                self.process_run_count += 1
                if self.process_run_count % CONFIG.main.idle_sleep_count == 0:
                    time.sleep(CONFIG.main.idle_sleep_second)

                    # process_run_count置0, 规避溢出
                    self.process_run_count = 0

            except Exception as e:
                self.logger.error(
                    f'monitor_proxy run error: {str(e)}, traceback.print_exc() is {traceback.format_exc()}')


if __name__=="__main__":
    from kaiwu_agent.utils.logging import ArenaLogger
    logger = ArenaLogger()
    monitor_proxy = MonitorProxy(logger)
    monitor_proxy.start()