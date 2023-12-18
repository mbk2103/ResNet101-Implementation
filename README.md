# ResNet101 Implementation

This repository contains an implementation of the ResNet101 architecture in PyTorch. The implementation includes a modular design with separate classes for the ResNet model, custom dataset handling, evaluation and visualization, and a training class.

## Table of Contents

- [Introduction](#introduction)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Evaluation and Visualization](#evaluation-and-visualization)
- [Training](#training)
- [Usage](#usage)
- [References](#references)

## Introduction

ResNet101 is a deep convolutional neural network architecture known for its success in image classification tasks. This implementation provides a clean and modular codebase for understanding and using the ResNet101 architecture.

## Architecture

The ResNet101 model is implemented using PyTorch. The architecture includes residual blocks, custom dataset handling, and evaluation and visualization components.

- **ResidualBlock**: Defines the building block for the ResNet model.
- **ResNet101v2**: Implements the overall ResNet101 architecture.

## Dataset

The implementation includes a custom dataset class for handling image data. You can easily replace the dataset with your own by modifying the `CustomDataset` class.

## Evaluation and Visualization

The `EvaluateVisualization` class provides methods for plotting loss curves and confusion matrices during model evaluation.

## Training

The `ResNetTrainer` class encapsulates the training process, making it easy to train the ResNet101 model on your dataset.

## Usage

To use this implementation, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/resnet101-implementation.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Modify the dataset path in the `main.py` file:

    ```python
    data_dir = "path/to/your/dataset"
    ```

4. Run the training script:

    ```bash
    python main.py
    ```

## References

This implementation is inspired by the ResNet architecture proposed in the paper:

- Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. ["Deep Residual Learning for Image Recognition"](https://arxiv.org/abs/1512.03385).

For additional insights, check out my Medium article on this implementation: [Unveiling the Power of ResNet101v2: A Deep Dive into Image Classification](https://medium.com/@mtburakk/unveiling-the-power-of-resnet101v2-a-deep-dive-into-image-classification-d1a10ad02f29)

Feel free to contribute to this repository or open issues if you encounter any problems.
