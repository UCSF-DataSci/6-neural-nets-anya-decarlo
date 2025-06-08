[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/7lg-zjaN)
[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=19731392)
# Assignment 6: Neural Networks and Health Data Analysis

## Overview

This assignment introduces neural networks and their applications in healthcare data analysis. You'll work through three parts:

1. **Basic Neural Networks**: Implement a simple neural network for EMNIST character recognition
2. **Convolutional Neural Networks**: Build a CNN for more complex image classification
3. **Time Series Analysis**: Apply neural networks to ECG signal classification

## Learning Objectives

- Implement and train neural networks using TensorFlow/Keras and/or Pytorch
- Apply CNNs for image classification
- Work with time series data using RNNs
- Evaluate model performance using appropriate metrics
- Interpret results in a healthcare context

## Setup

1. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Directory Structure**:

   ```
   datasci223_assignment6/
   ├── models/              # Saved models
   ├── results/            # Evaluation metrics
   │   ├── part_1/        # Part 1 results
   │   ├── part_2/        # Part 2 results
   │   └── part_3/        # Part 3 results
   ├── logs/              # Training logs
   ├── data/              # Downloaded datasets
   ├── part1_neural_networks_basics.md
   ├── part2_cnn_classification.md
   ├── part3_ecg_analysis.md
   └── requirements.txt
   ```

## Part 1: Neural Networks Basics

- Implement a simple neural network for EMNIST character recognition
- Use dense layers, activation functions, and dropout
- Save model as `models/emnist_classifier.keras`
- Save metrics in `results/part_1/emnist_classifier_metrics.txt`

Goals:
- Achieve > 80% accuracy on test set
- Minimize overfitting using dropout
- Train efficiently with appropriate batch size

## Part 2: CNN Classification

- Implement a CNN for EMNIST classification
- Choose between TensorFlow/Keras or PyTorch
- Save model as:
    - TensorFlow: `models/cnn_keras.keras`
    - PyTorch: `models/cnn_pytorch.pt` and `models/cnn_pytorch_arch.txt`
- Save metrics in `results/part_2/cnn_{framework}_metrics.txt`

Goals:
- Achieve > 85% accuracy on test set
- Minimize overfitting using batch normalization and dropout
- Train efficiently with appropriate batch size and learning rate

## Part 3: ECG Analysis

- Work with MIT-BIH Arrhythmia Database
- Choose between simple neural network or RNN/LSTM
- Save model as `models/ecg_classifier_{model_type}.keras`
- Save metrics in `results/part_3/ecg_classifier_{model_type}_metrics.txt`

Goals:
- Achieve > 75% accuracy on test set
- Achieve AUC > 0.80
- Achieve F1-score > 0.70
- Minimize overfitting using appropriate techniques
- Train efficiently with appropriate batch size

## Framework Options

1. **TensorFlow/Keras**:
   - Simpler syntax and more examples
   - Save models as `.keras` files

2. **PyTorch**:
   - Better for research and customization
   - Save models as `.pt` files with architecture in separate `.txt` file

## Common Issues and Solutions

1. **Data Loading**:
   - Problem: Dataset not found
   - Solution: Check directory structure and download scripts

2. **Model Training**:
   - Problem: Training instability
   - Solution: Use batch normalization, reduce learning rate
   - Problem: Overfitting
   - Solution: Increase dropout, use data augmentation

3. **Evaluation**:
   - Problem: Metrics format incorrect
   - Solution: Follow the exact format specified
   - Problem: Model performance below threshold
   - Solution: Adjust architecture, hyperparameters

## Resources

1. **Documentation**:
   - [TensorFlow Guide](https://www.tensorflow.org/guide)
   - [PyTorch Tutorials](https://pytorch.org/tutorials/)
   - [MIT-BIH Database](https://www.physionet.org/content/mitdb/1.0.0/)
   - [TensorFlow Neural Networks](https://www.tensorflow.org/tutorials)
   - [PyTorch CNNs](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
   - [RNN/LSTM Guide](https://www.tensorflow.org/guide/keras/rnn)
