# Mini-PyTorch in C

This project is a simplified version of PyTorch implemented in C, aiming to replicate essential tensor operations and functionalities, including tensor creation, broadcasting, matrix operations, and eventually, building a neural network.

## Table of Contents

- [Overview](#overview)
- [Tensor Creation Functions](#tensor-creation-functions)
- [Tensor Operations](#tensor-operations)
- [Autograd](#autograd)
- [Neural Network](#neural-network)

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

- **`tensor_tensor`**:  
  - Creates a new tensor from the provided raw data and shape information. 
  - **Parameters:**  
    - `data` (void*): A pointer to the raw input data (e.g., a multi-dimensional array).  
    - `shape` (int*): An array of integers specifying the dimensions of the tensor.  
    - `dims` (int): The number of dimensions of the tensor.  
  - **Returns:**  
    - `Tensor*`: A newly created tensor with the input data, shape, and metadata properly initialized.

## Tensor Operations

The following functions allow operations on tensors:

- **`tensor_reshape`**:  
  - Changes the shape of a tensor. If the data is contiguous, no copy is made; otherwise, it creates a contiguous copy of the data.  
  - **Parameters**:  
    - `input1` (Tensor*): Input tensor.  
    - `dim` (int): Number of dimensions for the new shape.  
    - `shape` (int*): Array specifying the new shape.  
  - **Returns**:  
    - `Tensor*`: A new tensor with the specified shape.

- **`tensor_broadcast`**:  
  - Broadcasts two tensors to a common shape according to PyTorch broadcasting semantics.  
  - **Parameters**:  
    - `input1` (Tensor*): First tensor.  
    - `input2` (Tensor*): Second tensor.  
  - **Returns**:  
    - `Tensor*`: A new tensor with the broadcasted shape.

- **`tensor_matmul`**:  
  - Performs a matrix multiplication. Supports both batch matrix multiplication and matrix-matrix products if both arguments are 2-dimensional.  
  - **Parameters**:  
    - `input1` (Tensor*): First input tensor.  
    - `input2` (Tensor*): Second input tensor.  
  - **Returns**:  
    - `Tensor*`: The result of the matrix multiplication.

- **`tensor_add`**:  
  - Adds two tensors element-wise. If their shapes do not match, it broadcasts them according to PyTorch broadcasting semantics.  
  - **Parameters**:  
    - `input1` (Tensor*): First input tensor.  
    - `input2` (Tensor*): Second input tensor.  
  - **Returns**:  
    - `Tensor*`: A tensor with the result of the addition.

- **`tensor_transpose`**:  
  - Swaps two specified dimensions of a tensor.  
  - **Parameters**:  
    - `input` (Tensor*): The tensor to transpose.  
    - `dim0` (int): The first dimension to swap.  
    - `dim1` (int): The second dimension to swap.  
  - **Returns**:  
    - `Tensor*`: A transposed tensor with the specified dimensions swapped.

- **`tensor_t`**:  
  - Transposes the last two dimensions of a tensor.  
  - **Parameters**:  
    - `input` (Tensor*): The tensor to transpose.  
  - **Returns**:  
    - `Tensor*`: A tensor with the last two dimensions transposed.

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

## Neural Network

This section describes the key components of building a neural network using the provided framework.

### Key Functions

- **`Linear`**:
  - Implements a fully connected (dense) layer.
  - **Parameters:**
    - `module` (Module*): The neural network module containing layer parameters.
    - `layernum` (int): The index of the layer in the module's parameter list.
    - `a` (Tensor*): The input tensor.
    - `in_features` (int): Number of input features.
    - `out_features` (int): Number of output features.
    - `use_bias` (int): Whether to add a bias term (1 for true, 0 for false).
  - **Returns**:
    - `Tensor*`: The result of the linear transformation.

- **`Relu`**:
  - Applies the ReLU (Rectified Linear Unit) activation function element-wise.
  - **Parameters:**
    - `a` (Tensor*): Input tensor.
  - **Returns**:
    - `Tensor*`: The tensor with ReLU applied.

- **`mse`**:
  - Computes the Mean Squared Error loss between predictions and labels.
  - **Parameters:**
    - `pred` (Tensor*): Predictions.
    - `label` (Tensor*): True labels.
  - **Returns**:
    - `Tensor*`: The computed MSE loss.

- **`zero_grad`**:
  - Resets the gradient buffers of all trainable parameters to zero.
  - **Parameters:**
    - `module` (Module*): The neural network module.
  - **Returns**:
    - `void`.

- **`optimizer_step`**:
  - Updates the parameters of the module using the specified optimizer type (e.g., SGD).
  - **Parameters:**
    - `module` (Module*): The neural network module.
    - `type` (char*): The optimizer type (e.g., \"sgd\").
  - **Returns**:
    - `void`.

### Example: Neural Network Implementation

Below is an example implementation of a neural network training process:

```c
Tensor *forward(Module *module, Tensor *a) {
    Tensor *layer = Linear(module, 0, a, a->shape[a->num_dims - 1], 5, 0);
    layer = Relu(layer);
    layer = Linear(module, 1, layer, layer->shape[layer->num_dims - 1], 1, 0);
    return layer;
}

int main() {
    tensor_set_seed(1337);
    CostHistory history;
    cost_history_init(&history);

    float data[20][2] = {
        {0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}, {0.7, 0.8}, {0.2, 0.3},
        {0.4, 0.6}, {0.6, 0.8}, {0.8, 0.9}, {0.1, 0.4}, {0.3, 0.5},
        {0.5, 0.7}, {0.7, 0.9}, {0.2, 0.5}, {0.4, 0.7}, {0.6, 0.9},
        {0.1, 0.3}, {0.3, 0.6}, {0.5, 0.8}, {0.7, 0.7}, {0.9, 0.8}
    };

    float labels[20] = {
        0.17, 0.37, 0.57, 0.77, 0.27,
        0.53, 0.73, 0.87, 0.30, 0.43,
        0.63, 0.83, 0.40, 0.60, 0.80,
        0.23, 0.50, 0.70, 0.70, 0.83
    };

    int data_shape[] = {1, 20, 2};
    Tensor *d = tensor_tensor(data, data_shape, 3);
    int label_shape[] = {20, 1};
    Tensor *l = tensor_tensor(labels, label_shape, 3);
    Module *module = nn();
    Tensor *prediction;

    for (int epoch = 0; epoch < 2000; epoch++) {
        printf(\"=========== epoch : %i =================\\n\", epoch);
        prediction = forward(module, d);
        if (epoch == 0) {
            printf(\"before training prediction\\n\");
            tensor_print(prediction);
        }

        Tensor *cost_tensor = mse(prediction, l);
        Tensor *cost = tensor_full(prediction->num_dims, prediction->shape, ((float *)cost_tensor->data)[0], 0);
        float current_cost = ((float *)cost_tensor->data)[0];
        if (epoch % 10 == 0) cost_history_add(&history, current_cost);

        tensor_backward(prediction, cost);
        optimizer_step(module, \"sgd\");
        zero_grad(module);
        free(cost);
        free(cost_tensor);
        free(prediction);
    }

    prediction = forward(module, d);
    tensor_print(prediction);
    plot_cost_ascii(&history);
}
```
Output
The program outputs the final predictions and a plot of the cost function over epochs:
```bash
Tensor of shape (1, 20, 1):
[[0.174256], [0.373380], [0.572504], ..., [0.821363]]

Cost Function Over Epochs
0.3333 ┐
|                                                           
|*                                                         
| **                                                       
|    ****************************************************** 
------------------------------------------------------------
0.0003 ┴────────────────────────────────────────────────── 200 epochs
        0           50          100         150         200         
```