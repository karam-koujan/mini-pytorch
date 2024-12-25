# Mini-PyTorch in C

This project is a simplified version of PyTorch implemented in C, aiming to replicate essential tensor operations and functionalities, including tensor creation, broadcasting, matrix operations, and eventually, building a neural network.

## Table of Contents

- [Overview](#overview)
- [Tensor Creation Functions](#tensor-creation-functions)
- [Tensor Operations](#tensor-operations)
- [Autograd](#autograd)

## Overview

This project implements a mini version of PyTorch using the C programming language. The primary focus is to handle tensor operations and implement key concepts needed for machine learning, such as automatic differentiation, matrix operations, broadcasting, and tensor manipulation. The goal is to build a system that can be expanded into a fully functional neural network framework.

## Tensor Creation Functions

The following functions allow the creation of tensors for different purposes:

- **`tensor_empty`**: 
  - Creates a tensor with an uninitialized data buffer. The tensor's shape is defined by the `dim` and `shape` parameters, but the data values are not initialized, meaning they may contain garbage values.
  - **Parameters:**
    - `dim` (int): The number of dimensions of the tensor.
    - `shape` (int*): An array specifying the size of each dimension.
  - **Returns**: 
    - `Tensor*`: A tensor with the specified shape but with uninitialized data.

- **`tensor_zeros`**: 
  - Creates a tensor filled with zeros. The tensor's shape is defined by the `dim` and `shape` parameters, and all elements are initialized to `0`.
  - **Parameters:**
    - `dim` (int): The number of dimensions of the tensor.
    - `shape` (int*): An array specifying the size of each dimension.
  - **Returns**:
    - `Tensor*`: A tensor with the specified shape filled with zeros.

- **`tensor_ones`**: 
  - Creates a tensor filled with ones. The tensor's shape is defined by the `dim` and `shape` parameters, and all elements are initialized to `1`.
  - **Parameters:**
    - `dim` (int): The number of dimensions of the tensor.
    - `shape` (int*): An array specifying the size of each dimension.
  - **Returns**:
    - `Tensor*`: A tensor with the specified shape filled with ones.

- **`tensor_rand`**: 
  - Creates a tensor filled with random values between 0 and 1. The random number generation is seeded using `srand()` to ensure variability across runs.
  - **Parameters:**
    - `dim` (int): The number of dimensions of the tensor.
    - `shape` (int*): An array specifying the size of each dimension.
  - **Returns**:
    - `Tensor*`: A tensor with the specified shape filled with random values between 0 and 1.

- **`tensor_full`**: 
  - Creates a tensor filled with a constant value (specified by the user). The tensor's shape is defined by the `dim` and `shape` parameters, and all elements are initialized to the specified constant value.
  - **Parameters:**
    - `dim` (int): The number of dimensions of the tensor.
    - `shape` (int*): An array specifying the size of each dimension.
    - `value` (float): The constant value used to fill the tensor.
  - **Returns**:
    - `Tensor*`: A tensor with the specified shape filled with the given constant value.

- **`tensor_print`**: 
  - Prints a given tensor to `stdout`. The function iterates over the tensor's data and prints it in a readable format.
  - **Parameters:**
    - `tensor` (Tensor*): The tensor to print.
  - **Returns**:
    - `void`: This function does not return any value. It simply prints the tensor to the standard output.

- **`tensor_set_require_grad`**: 
  - Sets the `requires_grad` flag for a given tensor. If `require_grad` is set to `1`, it initializes a gradient buffer for the tensor and marks it to track gradients. If `require_grad` is set to `0`, it frees the gradient buffer and disables gradient tracking.
  - **Parameters:**
    - `a` (Tensor*): The tensor whose `requires_grad` flag is being set.
    - `require_grad` (int): Flag indicating whether to enable (`1`) or disable (`0`) gradient tracking.

- **`tensor_detach`**: 
  - Creates a new tensor that shares the same data as the original tensor but does not track gradients. This is useful when you want to stop the tensor from contributing to the computational graph while retaining its data.
  - **Parameters:**
    - `a` (Tensor*): The tensor to detach.
  - **Returns**: 
    - `Tensor*`: A new tensor that shares the data of `a` but with `requires_grad` set to `0` and no gradient function.

- **`tensor_clone`**: 
  - Creates a new tensor that is a deep copy of the original tensor. The new tensor will have its own data (allocated memory) and preserve the original tensor's gradient information and metadata.
  - **Parameters:**
    - `a` (Tensor*): The tensor to clone.
  - **Returns**:
    - `Tensor*`: A new tensor that is a deep copy of `a`, with its own data and gradient metadata.


## Tensor Operations

The following functions allow operations on tensors:

- **`tensor_reshape(Tensor *input1, int dim, int *shape)`**: Returns a new tensor with new shape but it does not copy the original tensor data if data is contigious otherwise it create a contigious copy of the data.
- **`tensor_broadcast(Tensor *input1,Tensor *input2)`**: Returns a new broadcasted tensors following the pytorch broadcasting semantics.
- **`tensor_matmul(Tensor *input1,Tensor *input2)`**: Recreates the behavior of pytorch matmul, returning a batch matrix multiply, or matrix-matrix product if both arguments are two dimentional.
- **`tensor_add(Tensor *input1,Tensor *input2)`**: Preforms pairwise addition if shapes are not equals it broadcasts the tensor according to the pytorch broadcasting semantics.
- **`tensor_transpose(Tensor *input,dim0,dim1)`**: Returns a tensor that is a transposed version of input. The given dimensions dim0 and dim1 are swapped.
- **`tensor_t(Tensor *input)`**: It transpose the last two dimensions.

## Autograd

## Autograd: PyTorch-like Autograd Based on Backward Differentiation

This library implements a PyTorch-inspired automatic differentiation engine that supports operations like matrix multiplication (`matmul`). Gradients are computed using **backward mode autodifferentiation**, and tensors can be flagged with `require_grad` to track their gradients during computations.

### Example: Matrix Multiplication with Autograd

Below is an example of computing gradients for a matrix multiplication operation using the custom autograd system:

```c

    int a_shape[3] = {1, 2, 2};  // Shape of tensor `a`: 1 x 2 x 2
    int b_shape[3] = {1, 2, 3};  // Shape of tensor `b`: 1 x 2 x 3

    // Create tensors initialized with constant values
    Tensor *a = tensor_full(3, a_shape, 2.0, 0);  // Tensor a:
    /* [[2.0, 2.0], */
    /*  [2.0, 2.0]] */

    Tensor *b = tensor_full(3, b_shape, 3.0, 0);  // Tensor b:
    /* [[3.0, 3.0, 3.0], */
    /*  [3.0, 3.0, 3.0]] */

    // Enable gradient tracking
    tensor_set_require_grad(a, 1);
    tensor_set_require_grad(b, 1);

    // Perform matrix multiplication
    Tensor *c = tensor_matmul(a, b);
    /* [[12.0, 12.0, 12.0], */
    /*  [12.0, 12.0, 12.0]] */

    // Compute gradients
    tensor_backward(c, NULL);

    // Print tensors and their gradients
    tensor_print(a);  // Gradient of a:
    /* [[6.0, 6.0], */
    /*  [6.0, 6.0]] */

    tensor_print(b);  // Gradient of b:
    /* [[4.0, 4.0, 4.0], */
    /*  [4.0, 4.0, 4.0]] */
```
