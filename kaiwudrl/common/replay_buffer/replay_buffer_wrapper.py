#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file replay_buffer_wrapper.py
# @brief
# @author kaiwu
# @date 2023-11-28


import torch
import time
import threading
from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.common.utils.tf_utils import *


class ReverbDataset(torch.utils.data.IterableDataset):
    """
    主动采用reverb_client从reverb_server读取数据
    """

    def __init__(self, client):
        super().__init__()
        self.client = client
        self._table_names = ["{}_{}".format(CONFIG.reverb_table_name, i) for i in range(int(CONFIG.reverb_table_size))]

    def generate(self):
        while True:
            data = list(
                self.client.sample(
                    table=self._table_names[0],
                    num_samples=CONFIG.train_batch_size,
                    #    validation_timeout_ms=CONFIG.reverb_validation_timeout_ms,
                )
            )
            data = [x[0].data[0] for x in data]
            yield [data]

    def __iter__(self):
        # 从Reverb的服务器中获取数据
        return iter(self.generate())


class ReplayBufferWrapper(object):
    def __init__(self, tensor_names, tensor_dtypes, tensor_shapes, logger=None):
        self._tensor_names = tensor_names
        self._tensor_dtypes = tensor_dtypes
        self._tensor_shapes = tensor_shapes
        self._sorted_names = None
        self._sorted_dtypes = None
        self._sorted_shapes = None

        # 针对replaybuffer 统计信息
        self.proc_sample_cnt = 0
        self.skip_sample_cnt = 0

        self.logger = logger
        self.logger.info(f"train replaybuff, use {CONFIG.replay_buffer_type}")

    def init(self):
        if CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_REVERB:
            from kaiwudrl.common.replay_buffer.reverb_replay_buffer import (
                ReverbReplayBuffer,
            )

            self._replay_buffer = ReverbReplayBuffer(
                tuple([tf.TensorSpec(shape, dtype, name) for name, dtype, shape in zip(*self.sorted_tensor_spec())])
            )
            self._replay_buffer.init()

        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_TF_UNIFORM:
            self._replay_buffer = NotImplemented
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_ZMQ:
            from kaiwudrl.common.replay_buffer.zmq_replay_buffer import ZmqReplayBuffer

            self._replay_buffer = ZmqReplayBuffer(
                tuple([tf.TensorSpec(shape, dtype, name) for name, dtype, shape in zip(*self.sorted_tensor_spec())]),
                self.logger,
            )
        else:
            raise ValueError("ReplayBuffer currently only support reverb or tf_uniform or zmq!")

    def train_hooks(self, local_step_tensor=None):
        return []

    def sorted_tensor_spec(self):
        if self._sorted_names is None or self._sorted_dtypes is None or self._sorted_shapes is None:
            shapes = [
                tf.TensorShape(([int(CONFIG.rnn_time_steps)] + shape.dims[2:]) if CONFIG.use_rnn else shape.dims[1:])
                for shape in self._tensor_shapes
            ]
            tensor_infos = list(zip(self._tensor_names, self._tensor_dtypes, shapes))
            sorted_tensors_infos = sorted(tensor_infos, key=lambda x: x[0])
            tmp_uniq_names, names, dtypes, shapes = set(), [], [], []
            # uniq
            for item in sorted_tensors_infos:
                if item[0] not in tmp_uniq_names:
                    tmp_uniq_names.add(item[0])
                    names.append(item[0])
                    dtypes.append(item[1])
                    shapes.append(item[2])
            shapes = [
                tf.TensorShape(
                    [
                        1,
                    ]
                )
                if s.ndims == 0
                else s
                for s in shapes
            ]
            for i, (name, shape) in enumerate(list(zip(names, shapes))):
                only_keep_first = True if CONFIG.use_rnn and name in CONFIG.rnn_states else False
                if only_keep_first:
                    shape = tf.TensorShape([1] + shape.dims[1:])
                    shapes[i] = shape
                self.logger.info(f"train tensor spec: {name}, {shape}")

            # Replay buffer hooker needs `step` to filter expired samples.
            if CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_TF_UNIFORM:
                names += ["s"]
                dtypes += [tf.int64]
                shapes += [
                    tf.TensorShape(
                        [
                            1,
                        ]
                    )
                ]

            self._sorted_dtypes = dtypes
            self._sorted_names = names
            self._sorted_shapes = shapes

        return self._sorted_names, self._sorted_dtypes, self._sorted_shapes

    def dataset_from_generator(self):
        """
        dataset_from_generator
        1. reverb里, 采用dataset.from_generator来进行构造数据, 获取到具体数据, 再进行run_session
        2. zmq里, 共享内存的不做操作
        """
        if (
            CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_REVERB
            or CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_TF_UNIFORM
        ):
            if CONFIG.reverb_data_cache:
                dataset = self._replay_buffer.as_dataset_by_cache()
            else:
                dataset = self._replay_buffer.as_dataset()

            self._dataset_iter = tf.compat.v1.data.make_initializable_iterator(dataset)

            if CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_REVERB:
                next_tensors = self._dataset_iter.get_next()[1]
            elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_TF_UNIFORM:
                next_tensors = self._dataset_iter.get_next()
            else:
                assert False

            return next_tensors

        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_ZMQ:
            assert False

        else:
            assert False

    def dataset_from_generator_by_pytorch(self):
        """
        该方案从reverb里, 采用遍历方式获取到数据
        """
        if CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_REVERB:
            dataset = ReverbDataset(self._reverb_client)

            datas = next(iter(dataset))

            return datas

        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_ZMQ:
            return self._replay_buffer.next_by_batch_size()

        else:
            pass

    def input_tensors(self):
        """
        该方案是采用tf.compat.v1.placeholder_with_default占位符 + 业务自定义网络结构生成的流水线设计, 推荐
        """
        if CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_TF_UNIFORM:
            pass

        if CONFIG.reverb_data_cache:
            dataset = self._replay_buffer.as_dataset_by_cache()
        else:
            dataset = self._replay_buffer.as_dataset()

        self._dataset_iter = tf.compat.v1.data.make_initializable_iterator(dataset)
        if CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_REVERB:
            next_tensors = self._dataset_iter.get_next()[1]
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_TF_UNIFORM:
            next_tensors = self._dataset_iter.get_next()
        else:
            assert False

        tensors = [
            tf.compat.v1.placeholder_with_default(d, shape=[None, None] + d.get_shape().as_list()[2:])
            if CONFIG.use_rnn
            else tf.compat.v1.placeholder_with_default(d, shape=[None] + d.get_shape().as_list()[1:])
            for d in next_tensors
        ]

        return dict(zip(self._sorted_names, tensors))

    def extra_initializer_ops(self):
        return [self._dataset_iter.initializer]

    def extra_threads(self):
        if CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_REVERB:
            self._reverb_server = self._replay_buffer.build_reverb_server()
            self._reverb_client = self._replay_buffer.build_reverb_client()

            """
            server的wait是常驻线程
            """

            def start_reverb_server():
                self._reverb_server.wait()

            thread = threading.Thread(target=start_reverb_server)
            thread.daemon = True
            thread.start()

            def start_reverb_update_stats():
                has_inserted = False
                while True:
                    if TF_VERSION_MAJOR == 1:
                        if has_inserted and self.proc_sample_cnt == 0:
                            __ = self._reverb_client.server_stats_info(True)
                            has_inserted = False
                        else:
                            server_stats_info = self._reverb_client.server_stats_info(False)
                            ps_cnt, ss_cnt = 0, 0
                            for table_name in self._replay_buffer.table_names:
                                ps_cnt += server_stats_info[table_name].proc_frame_cnt
                                ss_cnt += server_stats_info[table_name].skip_frame_cnt
                            self.proc_sample_cnt = ps_cnt
                            self.skip_sample_cnt = ss_cnt
                            if not has_inserted and ps_cnt > 0:
                                has_inserted = True

                    # 复用 idle_sleep_second 参数
                    time.sleep(CONFIG.idle_sleep_second)

            # thread = threading.Thread(target=start_reverb_update_stats)
            # thread.daemon = True
            # thread.start()

        else:
            pass

    def reset(self, step, tf_sess):
        if CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_REVERB:
            self._replay_buffer.clear(self._reverb_client, step)
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_TF_UNIFORM:
            tf_sess.run(self._tf_replay_buffer_clear)
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_ZMQ:
            pass
        else:
            assert False

    def input_ready(self, tf_sess):
        if CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_REVERB:
            current_size = self._replay_buffer.total_size(self._reverb_client)
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_TF_UNIFORM:
            # 这里暂时没有实现
            current_size = tf_sess.run(self._tf_replay_buffer_total_size)
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_ZMQ:
            current_size = 0
        else:
            assert False

        # self.logger.debug(f"train current_size: {current_size}, CONFIG.train_batch_size: {CONFIG.train_batch_size}")
        return current_size >= int(CONFIG.train_batch_size)

    def add_sample(self, sample):
        """
        新增1条记录, 在zmq场景下使用, 即朝共享内存插入单条记录
        """
        if not sample:
            return False

        if CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_ZMQ:
            return self._replay_buffer.add_sample(sample)
        else:
            assert False

    # 获取样本接收速度
    def get_recv_speed(self):
        if CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_REVERB:
            return 0
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_TF_UNIFORM:
            return 0
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_ZMQ:
            return self._replay_buffer.get_insert_speed()
        else:
            assert False

    # 获取目前样本池里的数目
    def get_current_size(self):
        if CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_REVERB:
            return self._replay_buffer.total_size(self._reverb_client)
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_TF_UNIFORM:
            # 这里暂时没有实现
            return 0
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_ZMQ:
            return CONFIG.replay_buffer_capacity
        else:
            assert False

    # 获取目前样本池里插入的数目
    def get_insert_stats(self):
        if CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_REVERB:
            return self._replay_buffer.insert_stats(self._reverb_client)
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_ZMQ:
            return self._replay_buffer.get_insert_speed()
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_TF_UNIFORM:
            # 这里暂时没有实现
            return 0
        else:
            assert False
