#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file model_wrapper_pytorch.py
# @brief
# @author kaiwu
# @date 2023-11-28

from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine

if KaiwuDRLDefine.MODEL_PYTORCH == CONFIG.use_which_deep_learning_framework:
    from kaiwudrl.common.utils.torch_utils import *

from kaiwudrl.common.checkpoint.model_file_common import (
    before_save_model,
    after_save_model,
    update_id_list,
    check_path_id_valid,
    find_id_in_id_list,
)
import glob
import os
import time
import traceback


class StandardModelWrapperPytorch:
    """
    StandardModelWrapperPytorch类, actor和learner都会使用, 主要用于预测, 训练等
    """

    def __init__(self, model, logger, server=None) -> None:
        self.model = model
        self.logger = logger

        # 统计值
        self.train_count = 0
        self.predict_count = 0

        # 按照频率来保存控制参数
        self.save_model_count = 0
        # 按照总数来控制参数
        self.save_model_all_count = 0

        if CONFIG.svr_name == KaiwuDRLDefine.SERVER_LEARNER:
            self.local_rank = 0

        # 主learner
        self.is_chief = CONFIG.svr_name == KaiwuDRLDefine.SERVER_LEARNER and self.local_rank == 0

        # 最后一次保存模型的时间
        self.last_save_model_time = 0

        # 进程启动时间
        self.process_start_time = time.monotonic()

        # 给业务设置下日志接口
        self.set_logger()

        # 因为pytorch需要用户确保保存最多多少model文件
        self.file_queue = []

    def should_stop(self):
        # 需要业务提供下该方法
        # return self.model.should_stop()
        return False

    def set_logger(self):
        # 由于已经在init时传递了logger对象, 故这里不需要再传递
        if hasattr(self.model, "set_logger"):
            self.model.set_logger(self.logger)

    def close(self):
        if hasattr(self.model, "stop"):
            return self.model.stop()

    def before_train(self):
        pass

    def add_file_to_queue(self):
        id = self.train_count

        # 采用模糊匹配的方法来操作
        model_file_names = glob.glob(
            f"{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}/{KaiwuDRLDefine.KAIWU_MODEL_CKPT}-{id}.*"
        )
        if not model_file_names:
            return

        self.file_queue.append(model_file_names)
        if len(self.file_queue) >= CONFIG.max_save_model_file_count:
            model_file_names_to_delete = self.file_queue.pop(0)
            if model_file_names_to_delete:
                for to_model_file_name in model_file_names_to_delete:
                    if os.path.exists(to_model_file_name):
                        os.remove(to_model_file_name)

    def after_train(self):
        # 本次是否执行了更新model文件的操作
        has_model_file_changed = False
        self.train_count += 1
        if self.train_count % CONFIG.dump_model_freq == 0:
            if getattr(CONFIG, f"{CONFIG.svr_name}_device_type") == KaiwuDRLDefine.MACHINE_DEVICE_NPU:
                torch.npu.set_device(torch.device(self.model.model.device))

            self.model.save_model(
                path=f"{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}/",
                id=self.train_count,
                source=KaiwuDRLDefine.SAVE_OR_LOAD_MODEL_By_FRAMEWORK,
            )

            # 放入队列控制占用大小以免磁盘无限增加被驱逐
            self.add_file_to_queue()

            # 维护id_list列表
            update_id_list(self.train_count, framework=True)

            has_model_file_changed = True

        return has_model_file_changed, self.train_count

    def before_save_param(self):
        # 保存模型前的操作
        before_save_model()

    def after_save_param(self, id):
        """
        业务侧调用生成model文件后, 需要做下面工作:
        1. 生成json文件
        2. 生成tar.gz文件
        3. 清空类似下面的文件/data/user_ckpt_dir/gorge_walk_dp, 这样会导致该目录下只是保存最新的model文件, 历史的采用tar.gz包放置
        """
        after_save_model(self.process_start_time, id)

    def do_save_param(self, func, agent, path, id):
        """
        保存模型文件
        """

        # 保存模型前的操作
        self.before_save_param()

        # 调用的是业务的func
        func(agent, path=path, id=id)

        # 保存模型后的操作
        self.after_save_param(id)

        self.save_model_count += 1
        self.save_model_all_count += 1

    def save_param_by_source(self, path=None, id=None, source=KaiwuDRLDefine.SAVE_OR_LOAD_MODEL_By_FRAMEWORK):
        """
        该函数是直接从kaiwu_rl_helper_standard、on_policy_trainer调用进来的, 即KaiwuDRL调用
        若 source="sigterm",则用于处理优雅退出
        """
        if getattr(CONFIG, f"{CONFIG.svr_name}_device_type") == KaiwuDRLDefine.MACHINE_DEVICE_NPU:
            torch.npu.set_device(torch.device(self.model.model.device))

        self.model.save_model(path=path, id=id, source=source)

    def save_param(
        self,
        agent,
        func,
        *args,
        source=KaiwuDRLDefine.SAVE_OR_LOAD_MODEL_By_USER,
        **kargs,
    ):
        """
        该函数是从使用者调用进来的, 即使用者调用
        """
        # id取值为self.train_count
        kargs["id"] = self.train_count

        self.save_param_detail(agent, func, *args, source=source, **kargs)

    def save_param_detail(
        self,
        agent,
        func,
        source=KaiwuDRLDefine.SAVE_OR_LOAD_MODEL_By_USER,
        path=None,
        id="1",
    ):
        """
        默认是业务调用
        如果是框架调用, 需要传递source="framework"参数, 如下规则:
        1. 无限制
        如果是SIGTERM调用, 需要传递source="sigterm"参数, 如下规则:
        1. 无限制
        如果是业务调用, 如下规则:
        1. 用户保存模型最大次数, 设置为小于等于0代表不限制
        2. 用户保存模型的频率, 设置为小于等于0代表不限制
        3. 其保存模型的path不能用KaiwuDRL默认保存model文件的path, 以免混淆
        """
        try:
            if source == KaiwuDRLDefine.SAVE_OR_LOAD_MODEL_BY_SIGTERM:
                # 非框架调用下默认的保存目录和框架已经有的文件目录不一样
                path = f"{CONFIG.user_ckpt_dir}/{CONFIG.app}_{CONFIG.algo}"

                # 优雅退出时如果id为0, 即一步也没有训练成功则不需要保存模型文件, 该模型文件也是随机的
                if not id:
                    self.logger.info(f"train_step is 0, so not save_model")
                else:
                    # 优雅退出时主动调用一次用户侧保存模型函数
                    if not find_id_in_id_list(id, framework=False):
                        self.do_save_param(func, agent, path, id)
                    else:
                        self.logger.info(f"{KaiwuDRLDefine.KAIWU_MODEL_CKPT}-{id} already exists")

            elif source == KaiwuDRLDefine.SAVE_OR_LOAD_MODEL_By_FRAMEWORK:
                # 非框架调用下默认的保存目录和框架已经有的文件目录不一样
                path = f"{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}"

                # 调用的是业务的func
                func(agent, path=path, id=id)

            elif source == KaiwuDRLDefine.SAVE_OR_LOAD_MODEL_By_USER:
                # 非框架调用下默认的保存目录和框架已经有的文件目录不一样
                path = f"{CONFIG.user_ckpt_dir}/{CONFIG.app}_{CONFIG.algo}"

                if CONFIG.user_save_mode_max_count > 0:
                    if self.save_model_all_count >= CONFIG.user_save_mode_max_count:
                        self.logger.error(
                            f" save_param() self.save_model_all_count {self.save_model_all_count} "
                            f"> CONFIG.user_save_mode_max_count {CONFIG.user_save_mode_max_count}, "
                            f"please check your code for any error"
                        )

                        # 在不保存文件的基础上还需要增加该计数, 目的是为用户提示
                        self.save_model_all_count += 1
                        return

                if CONFIG.user_save_model_max_frequency_per_min > 0:
                    # 获取当前时间
                    current_time = time.time()
                    if current_time - self.last_save_model_time >= 60:
                        self.save_model_count = 0
                        self.last_save_model_time = current_time

                    if self.save_model_count <= CONFIG.user_save_model_max_frequency_per_min:
                        self.do_save_param(func, agent, path, id)
                    else:
                        self.logger.error(
                            f" save_param() user_save_model_max_frequency_per_min > "
                            f"CONFIG.user_save_model_max_frequency_per_min "
                            f"{CONFIG.user_save_model_max_frequency_per_min}, so return "
                        )
                else:
                    self.do_save_param(func, agent, path, id)

            else:
                # 未来扩展
                pass

        except RuntimeError:
            self.logger.error(
                f" save_param() RuntimeError Exception, traceback.print_exc() is {traceback.format_exc()}"
            )
            return

        except Exception as e:
            self.logger.error(f" save_param() Exception {str(e)}, traceback.print_exc() is {traceback.format_exc()}")
            return

    def before_predict(self, predict_data):
        return isinstance(predict_data, dict)

    def after_predict(self, batch_size):
        self.predict_count += batch_size

    # train函数, 单机单进程版本调用
    def train_local(self, data, extra_tensors=None):
        try:
            self.before_train()

            # 具体的训练流程
            values = self.model.learn(data, train=True)

            # 返回是否更新了model文件, 更新的model文件的ID
            has_model_file_changed, model_file_id = self.after_train()

            return values, has_model_file_changed, model_file_id

        except RuntimeError:
            self.logger.error(
                f" train_local() RuntimeError Exception, traceback.print_exc() is {traceback.format_exc()}"
            )
            return None, None, None

        except Exception as e:
            self.logger.error(f" train_local() Exception {str(e)}, traceback.print_exc() is {traceback.format_exc()}")
            return None, None, None

    # train 函数, 集群版本调用
    def train(self, extra_tensors=None):
        try:
            self.before_train()

            # 具体的训练流程
            data = self.get_data_from_reverb()
            if not data:
                return None, False, -1

            values = self.model.learn(data, train=True)

            # 返回是否更新了model文件, 更新的model文件的ID
            has_model_file_changed, model_file_id = self.after_train()

            return values, has_model_file_changed, model_file_id

        except RuntimeError:
            self.logger.error(f" train() RuntimeError Exception, traceback.print_exc() is {traceback.format_exc()}")
            return None, None, None

        except Exception as e:
            self.logger.error(f" train() Exception {str(e)}, traceback.print_exc() is {traceback.format_exc()}")
            return None, None, None

    # exploit函数, 单机单进程版本使用
    def exploit_local(self, predict_data, predict=False):
        try:
            values = self.model.exploit(predict_data, predict=predict)
            return values

        except RuntimeError:
            self.logger.error(
                f" exploit_local() RuntimeError Exception, traceback.print_exc() is {traceback.format_exc()}"
            )
            return None

        except Exception as e:
            self.logger.error(f" exploit_local() Exception {str(e)}, traceback.print_exc() is {traceback.format_exc()}")
            return None

    # predict函数, 单机单进程版本使用
    def predict_local(self, predict_data, predict=False):
        try:
            # 具体的预测流程
            self.before_predict(predict_data)

            # 部分场景需要更新predict_count
            if hasattr(self.model, "update_predict_count"):
                self.model.update_predict_count(self.predict_count)

            values = self.model.predict(predict_data, predict=predict)

            self.after_predict(len(predict_data))

            return values

        except RuntimeError:
            self.logger.error(
                f" predict_local() RuntimeError Exception, traceback.print_exc() is {traceback.format_exc()}"
            )
            return None

        except Exception as e:
            self.logger.error(f" predict_local() Exception {str(e)}, traceback.print_exc() is {traceback.format_exc()}")
            return None

    # exploit函数, 集群使用
    def exploit(self, predict_data):
        try:
            values = self.model.exploit(predict_data, predict=True)
            return values

        except RuntimeError:
            self.logger.error(f" exploit() RuntimeError Exception, traceback.print_exc() is {traceback.format_exc()}")
            return None

        except Exception as e:
            self.logger.error(f" exploit() Exception {str(e)}, traceback.print_exc() is {traceback.format_exc()}")
            return None

    # predict函数, 集群使用
    def predict(self, predict_data, batch_size):
        try:
            # 具体的预测流程
            self.before_predict(predict_data)

            # 部分场景需要更新predict_count
            if hasattr(self.model, "update_predict_count"):
                self.model.update_predict_count(self.predict_count)

            values = self.model.predict(predict_data, predict=True)

            self.after_predict(batch_size)

            return values

        except RuntimeError:
            self.logger.error(f" predict() RuntimeError Exception, traceback.print_exc() is {traceback.format_exc()}")
            return None

        except Exception as e:
            self.logger.error(f" predict() Exception {str(e)}, traceback.print_exc() is {traceback.format_exc()}")
            return None

    def get_global_step(self):
        return self.train_count

    @property
    def train_stat(self):
        return self.train_count

    @property
    def predict_stat(self):
        return self.predict_count

    @property
    def name(self):
        return "StandardModelWrapperPytorch"

    @property
    def tf_sess(self):
        return self.sess

    def standard_load_model(self, path, id):
        try:
            # 调用业务的load_model, 带上load_model字样
            self.model.load_model(path=path, id=id, load_model=True)
            return True

        except RuntimeError:
            self.logger.error(
                f" standard_load_model() RuntimeError Exception, traceback.print_exc() is {traceback.format_exc()}"
            )
            return False

        except Exception as e:
            self.logger.error(
                f" standard_load_model() Exception {str(e)}, traceback.print_exc() is {traceback.format_exc()}"
            )
            return False

    # 加载model文件, 目前只有评估模式才会使用, 训练模式修改为用户来主动加载model文件
    def standard_load_last_new_model_by_framework(self, path, id, framework=False):
        if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODEL_EVAL:
            # 判断参数是否合法
            if not check_path_id_valid(path, id):
                self.logger.error(
                    f"standard_load_last_new_model_by_framework from models_path {path}, id {id} failed, please check"
                )
                return False

            # 直接调用业务侧的load_model
            self.model.load_model(path=path, id=id, load_model=True)
            return True
        else:
            return False

    # 直接调用业务类的standard_load_last_new_model
    def standard_load_last_new_model(self, agent, func, path=None, id="1"):
        try:
            if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODEL_TRAIN:
                """
                需要按照用户指定的ID就加载哪个ID, 不需要指定
                """
                func(agent, path, id)

                self.logger.info(
                    f"train mode predict standard_load_last_new_model from models_path {path}, "
                    f"checkpoint_id {id} success"
                )

            elif CONFIG.run_mode == KaiwuDRLDefine.RUN_MODEL_EVAL:
                func(agent, path, id)

                self.logger.info(
                    f"eval mode predict standard_load_last_new_model from {path}, checkpoint_id {id} success"
                )

            else:
                pass

        except RuntimeError:
            self.logger.error(
                f" standard_load_last_new_model() RuntimeError Exception, "
                f"traceback.print_exc() is {traceback.format_exc()}"
            )
            return

        except Exception as e:
            self.logger.error(
                f" standard_load_last_new_model() Exception {str(e)}, "
                f"traceback.print_exc() is {traceback.format_exc()}"
            )
            return

    def preload_model_file(self, preload_model_dir, preload_model_id):
        """
        预加载模型文件, 直接调用业务类, 步骤如下:
        1. 不需要清空以前的checkpoint文件, 因为以前的checkpoint文件会被很快覆盖掉
        2. 调用业务类的load_model
        3. 调用业务类的save_model
        """
        if not check_path_id_valid(preload_model_dir, preload_model_id):
            self.logger.error(
                f"preload_model_file failed, but preload_model_dir {preload_model_dir} or "
                f"preload_model_id {preload_model_id} not valid, please check"
            )
            return

        try:
            # 调用业务的load_model
            self.model.load_model(path=preload_model_dir, id=preload_model_id)
            self.train_count = preload_model_id

            self.logger.info(f" preload_model_file success, path is {preload_model_dir}, id is {preload_model_id}")
        except Exception as e:
            self.logger.error(
                f" preload_model_file() Exception {str(e)}, traceback.print_exc() is {traceback.format_exc()}"
            )

    def set_dataset(self, replay_buffer_wrapper):
        self.replay_buffer_wrapper = replay_buffer_wrapper

    def is_chief(self):
        return self.is_chief

    def get_model_object(self):
        return self.model

    def get_data_from_reverb(self):
        # 采用pytorch方法从reverb获取数据
        return self.replay_buffer_wrapper.dataset_from_generator_by_pytorch()
