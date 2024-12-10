# Mini-PyTorch in C

This project is a simplified version of PyTorch implemented in C, aiming to replicate essential tensor operations and functionalities, including tensor creation, broadcasting, matrix operations, and eventually, building a neural network.

## Table of Contents

- [Overview](#overview)
- [Tensor Creation Functions](#tensor-creation-functions)
-[Tensor Operations](#tensor-operations)

## Overview

This project implements a mini version of PyTorch using the C programming language. The primary focus is to handle tensor operations and implement key concepts needed for machine learning, such as automatic differentiation, matrix operations, broadcasting, and tensor manipulation. The goal is to build a system that can be expanded into a fully functional neural network framework.

## Tensor Creation Functions

The following functions allow the creation of tensors for different purposes:

- **`tensor_empty`**: Creates a tensor with an uninitialized data buffer.
- **`tensor_zeros`**: Creates a tensor filled with zeros.
- **`tensor_ones`**: Creates a tensor filled with ones.
- **`tensor_rand`**: Creates a tensor filled with random values between 0 and 1. The random number generation is seeded using `srand()` to ensure variability across runs.
- **`tensor_full`**: Creates a tensor filled with a constant value (specified by the user).
- **`tensor_print`**: prints a given tensor to stdout.

## Tensor Operations

The following functions allow operations on tensors:

- **`tensor_reshape`**: returns a new tensor with new shape but it does not copy the original tensor data.
- **`tensor_broadcast`**: returns a new broadcasted tensors following the pytorch broadcasting semantics.
- **`tensor_matmul`**: recreates the behavior of pytorch matmul, returning a batch matrix multiply, or matrix-matrix product if both arguments are two dimentional.
- **`tensor_add`**: preforms pairwise addition if shapes are not equals it broadcasts the tensor according to the pytorch broadcasting semantics.



