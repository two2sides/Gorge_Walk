import setuptools
from setuptools import setup
import glob

from kaiwudrl.common.pybind11.zmq_ops.common import get_ext_suffix

print(get_ext_suffix())

setup(
    name="zmq_ops",
    packages=["zmq_ops"],
    version="0.1.0",
    description="TensorFlow ZMQ Ops",
    author="Yuxin Wu",
    zip_safe=False,
    include_package_data=True,
    package_data={"zmq_ops": ["*" + get_ext_suffix()]},
    install_requires=[],
)
