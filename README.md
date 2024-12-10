# Mini-PyTorch in C

This project is a simplified version of PyTorch implemented in C, aiming to replicate essential tensor operations and functionalities, including tensor creation, broadcasting, matrix operations, and eventually, building a neural network.

## Table of Contents

- [Overview](#overview)
- [Tensor Operations](#tensor-operations)
  - [Tensor Creation Functions](#tensor-creation-functions)
  - [Tensor Reshaping](#tensor-reshaping)
  - [Tensor Broadcasting](#tensor-broadcasting)
  - [Tensor Operations](#tensor-operations)
- [Backpropagation and Gradients](#backpropagation-and-gradients)
- [Neural Network](#neural-network)
- [Usage Examples](#usage-examples)

## Overview

This project implements a mini version of PyTorch using the C programming language. The primary focus is to handle tensor operations and implement key concepts needed for machine learning, such as automatic differentiation, matrix operations, broadcasting, and tensor manipulation. The goal is to build a system that can be expanded into a fully functional neural network framework.

## Tensor Operations

### Tensor Creation Functions

The following functions allow the creation of tensors for different purposes:

- **`tensor_empty`**: Creates a tensor with an uninitialized data buffer.
- **`tensor_zeros`**: Creates a tensor filled with zeros.
- **`tensor_rand`**: Creates a tensor filled with random values between 0 and 1. The random number generation is seeded using `srand()` to ensure variability across runs.
- **`tensor_full`**: Creates a tensor filled with a constant value (specified by the user).

