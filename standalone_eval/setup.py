#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="cc_eval",
    version="0.0.1",
    description="Describe Your Cool Project",
    author="",
    author_email="https://github.com/chunchet-ng/icpr_lowlight/tree/clean/standalone_eval",
    url="https://github.com/user/project",
    install_requires=[
        "natsort",
        "scikit-image",
        "loguru",
        "easydict",
        "Polygon3",
        "lpips",
        "pyiqa",
    ],
    packages=find_packages(),
)
