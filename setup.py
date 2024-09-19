#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="glioma_segmentation",
    version="0.1.2",
    description="Segmentation of adult gliomas post-treatment using MONAI and U-Net with PyTorch Lightning and Hydra",
    author="Marcel Musialek",
    author_email="mnem12321@gmail.com",
    url="https://github.com/EverScaleNet/ES-Segmentation-Adult-Glioma",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
        ]
    },
)
