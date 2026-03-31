from setuptools import setup
import os
from glob import glob

package_name = "cyberzoo_dataset"

models = [
    "models/virtual_drone_cam/model.sdf",
    "models/virtual_drone_cam/model.config",
]

launch_files = [
    "launch/generate_cyberzoo_dataset.launch.py",
]

data_files = [
    ("share/ament_index/resource_index/packages",
     ["resource/" + package_name]),
    ("share/" + package_name, ["package.xml"]),
    ("share/" + package_name + "/launch", launch_files),
    ("share/" + package_name + "/models/virtual_drone_cam", models),
]

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=data_files,
    install_requires=[
        "setuptools",
    ],
    zip_safe=True,
    maintainer="AMAV User",
    maintainer_email="you@example.com",
    description="Cyberzoo dataset generator using Gazebo and ROS 2.",
    license="BSD-3-Clause",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "generate_cyberzoo_dataset = cyberzoo_dataset.generate_cyberzoo_dataset:main",
        ],
    },
)

