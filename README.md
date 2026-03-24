# Deep Learning Framework from Scratch

A lightweight, layer-oriented deep learning framework built from scratch using NumPy. This project extends the basic neural network framework by adding convolutional layers, pooling, advanced initializers, and momentum-based optimizers.

## Overview

This framework builds upon the basic neural network implementation (BasicNN) by introducing architectural components essential for computer vision tasks. It follows the same layer-oriented architecture where each layer implements `forward()` and `backward()` operations. New additions include:

- **Initializers**: Xavier and He initialization for improved convergence
- **Advanced Optimizers**: SGD with Momentum and Adam
- **Convolutional Layers**: 1D and 2D convolutions with stride and same padding
- **Pooling**: Max pooling for dimensionality reduction
- **Flatten**: Bridge between convolutional and fully connected layers
