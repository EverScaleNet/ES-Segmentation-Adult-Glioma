# ES-Segmentation-Adult-Glioma

This repository contains the implementation of a segmentation model for post-treatment adult gliomas using the U-Net architecture. The project leverages MONAI, PyTorch Lightning, Hydra, and wandb for efficient model development and experimentation.

## Project Overview

The goal of this project is to develop a robust model for segmenting post-treatment gliomas from MRI scans. We utilize state-of-the-art tools and libraries to ensure high performance and reproducibility.

## Directory Structure

```plaintext
ES-Segmentation-Adult-Glioma/
├── configs/                 # Hydra configuration files
│   ├── config.yaml
│   ├── dataset.yaml
│   ├── model.yaml
│   └── trainer.yaml
│
├── data/                    # Data directory
│   ├── raw/                 # Raw data
│   └── processed/           # Processed data
│
├── src/                     # Source code
│   ├── data/                # Data processing scripts
│   ├── models/              # Model definitions
│   └── utils/               # Utility scripts
│ 
├── scripts/                 # Auxiliary scripts
│
├── tests/                   # Unit and integration tests
│
├── environment.yml          # Conda environment configuration
└── requirements.txt         # Additional pip dependencies
```


## Getting Started

### Prerequisites

- Conda (Miniconda or Anaconda)
- Python 3.10

### Setup

1. **Clone the repository:**
    ```bash
    git clone https://github.com/EverScaleNet/ES-Segmentation-Adult-Glioma.git
    cd ES-Segmentation-Adult-Glioma
    ```

2. **Create and activate the Conda environment:**
    ```bash
    conda create -n glioma-segmentation
    conda activate glioma-segmentation
    ```

3. **Install additional dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Project Components

### MONAI
MONAI (Medical Open Network for AI) is a PyTorch-based framework for deep learning in healthcare imaging, optimized for high performance and ease of use.

### PyTorch Lightning
PyTorch Lightning provides a high-level interface for PyTorch, allowing for more readable and flexible code, and simplifying the process of training models on multiple GPUs.

### Hydra
Hydra is a framework for managing configuration files, making it easy to compose and override configurations, which is particularly useful for experiments and hyperparameter tuning.

### Weights and Biases (wandb)
wandb is used for tracking experiments, visualizing results, and managing hyperparameters, providing a comprehensive suite for ML experiment management.

## Acknowledgements

This project was developed using the Lightning-Hydra-Template and data comes from the BraTS 2024 Challenge.

## Contact

For any questions or issues, please contact EverScaleNet.
