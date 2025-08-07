'''
Author: Haotian Liang haotianliang10@gmail.com
Date: 2025-08-05 15:09:09
LastEditors: Haotian Liang haotianliang10@gmail.com
LastEditTime: 2025-08-05 15:09:31
'''
from setuptools import setup, find_packages

setup(
    name="serl_robot_infra",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "pyrealsense2",
        "pymodbus==2.5.3",
        "opencv-python",
        "pyquaternion",
        "pyspacemouse",
        "hidapi",
        "pyyaml",
        "scipy",
        "defusedxml",
    ],
)
