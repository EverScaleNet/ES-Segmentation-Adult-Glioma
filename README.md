


<div align="center">

# Your Project Name

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

This repository contains the implementation of a segmentation model for post-treatment adult gliomas using the U-Net architecture. The project leverages MONAI, PyTorch Lightning, Hydra, and wandb for efficient model development and experimentation.

The goal of this project is to develop a robust model for segmenting post-treatment gliomas from MRI scans. We utilize state-of-the-art tools and libraries to ensure high performance and reproducibility.

This project was developed using the Lightning-Hydra-Template and data comes from the BraTS 2024 Challenge.

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/EverScaleNet/ES-Segmentation-Adult-Glioma.git
cd ES-Segmentation-Adult-Glioma

# [OPTIONAL] create conda environment
conda create -n glioma-segmentation
conda activate glioma-segmentation

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt

# or install project 
or pip install -e .
```

## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```
## Contact

For any questions or issues, please contact EverScaleNet.

## Project Components

### MONAI
MONAI (Medical Open Network for AI) is a PyTorch-based framework for deep learning in healthcare imaging, optimized for high performance and ease of use.

### PyTorch Lightning
PyTorch Lightning provides a high-level interface for PyTorch, allowing for more readable and flexible code, and simplifying the process of training models on multiple GPUs.

### Hydra
Hydra is a framework for managing configuration files, making it easy to compose and override configurations, which is particularly useful for experiments and hyperparameter tuning.

### Weights and Biases (wandb)
wandb is used for tracking experiments, visualizing results, and managing hyperparameters, providing a comprehensive suite for ML experiment management.

## Acknowledgements(once again)
data comes from the BraTS adult glioma 2024 Challenge.
This project was developed using the Lightning-Hydra-Template

## Contact

For any questions or issues, please contact EverScaleNet.