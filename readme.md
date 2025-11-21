# 🚀 Mini-PyTorch: A Deep Learning Library in Pure C

![C](https://img.shields.io/badge/Language-C-blue.svg)
![Build](https://img.shields.io/badge/Build-Makefile-lightgrey.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Mini-PyTorch** is a compact, educational tensor library written from scratch in pure C. Inspired by the core functionalities of PyTorch, it provides a dynamic Tensor object, a powerful automatic differentiation engine (autograd), and the essential building blocks for creating and training neural networks.


## Table of Contents

-   [Features](#features)
-   [How to Build](#how-to-build)
-   [API Documentation & Examples](#api-documentation--examples)
    -   [1. Tensor Creation](#1-tensor-creation)
        -   [`tensor_zeros`](#tensor_zeros)
        -   [`tensor_ones`](#tensor_ones)
        -   [`tensor_full`](#tensor_full)
        -   [`tensor_urand`](#tensor_urand)
        -   [`tensor_from_arr`](#tensor_from_arr)
    -   [2. Tensor Operations](#2-tensor-operations)
        -   [Mathematical Operations](#mathematical-operations)
        -   [Manipulation Operations](#manipulation-operations)
    -   [3. Autograd Engine](#3-autograd-engine)
        -   [`tensor_set_require_grad`](#tensor_set_require_grad)
        -   [`tensor_backward`](#tensor_backward)
    -   [4. Neural Network (nn) Module](#4-neural-network-nn-module)
        -   [`module_constructor`](#module_constructor)
        -   [`Linear`](#linear)
        -   [`tensor_relu`](#tensor_relu)
        -   [`optimizer_step`](#optimizer_step)
        -   [`module_zero_grad`](#module_zero_grad)
        -   [`module_free`](#module_free)
-   [Putting It All Together: A Complete Example](#putting-it-all-together-a-complete-example)

## ✨ Features

-   🧠 **Dynamic Tensors**: Create and manipulate multi-dimensional arrays with support for various data types (`float32`, `float64`, `int32`, `int64`).
-   📡 **Automatic Broadcasting**: Tensors of different shapes are automatically expanded for element-wise operations, just like in NumPy and PyTorch.
-   🔄 **Autograd Engine**: A powerful automatic differentiation engine that tracks operations to compute gradients for any sequence of tensor computations.
-   🤖 **Neural Network Primitives**: High-level building blocks like `Linear` layers and `ReLU` activation functions to construct simple models with ease.

## 🛠️ How to Build

To build the library and run the main executable, use the provided `Makefile`. You'll need a C compiler like `gcc` or `clang`.

```bash
# Compile the library and create the executable 'mini_pytorch'
make

# Run the example neural network training
./mini_pytorch

# Clean up object files
make clean

# Clean up object files and the executable
make fclean
```

## 📚 API Documentation & Examples

### 1. Tensor Creation

Create new tensors from scratch or from existing data.

#### `tensor_zeros`

Creates a tensor of a given shape filled with zeros.
```c
Tensor *tensor_zeros(const int64_t *shape, int64_t ndim, Dtype type, Device device);
```
**Example:**
```c
int64_t shape[] = {2, 3};
Tensor *t = tensor_zeros(shape, 2, FLOAT32, CPU);
tensor_print(t);
// Output:
// Tensor of shape (2,3):
// [0.00,0.00,0.00],
// [0.00,0.00,0.00]
```

#### `tensor_ones`

Creates a tensor of a given shape filled with ones.
```c
Tensor *tensor_ones(const int64_t *shape, int64_t ndim, Dtype type, Device device);
```
**Example:**
```c
int64_t shape[] = {2, 3};
Tensor *t = tensor_ones(shape, 2, FLOAT32, CPU);
tensor_print(t);
// Output:
// Tensor of shape (2,3):
// [1.00,1.00,1.00],
// [1.00,1.00,1.00]
```

#### `tensor_full`

Creates a tensor of a given shape filled with a specified scalar value.
```c
Tensor *tensor_full(const int64_t *shape, int64_t ndim, Dtype type, Device device, void *val);
```
**Example:**
```c
int64_t shape[] = {2, 3};
float val = 7.5f;
Tensor *t = tensor_full(shape, 2, FLOAT32, CPU, &val);
tensor_print(t);
// Output:
// Tensor of shape (2,3):
// [7.50,7.50,7.50],
// [7.50,7.50,7.50]
```

#### `tensor_urand`

Creates a tensor with random values uniformly distributed between a `min` and `max`.
```c
Tensor *tensor_urand(const int64_t *shape, int64_t ndim, Dtype type, Device device, double min, double max);
```
**Example:**
```c
tensor_set_seed(1337); // For reproducibility
int64_t shape[] = {2, 2};
Tensor *t = tensor_urand(shape, 2, FLOAT32, CPU, -10.0, 10.0);
tensor_print(t);
```

#### `tensor_from_arr`

Creates a tensor from an existing C array. The data is copied.
```c
Tensor *tensor_from_arr(void *arr, const int64_t *shape, int64_t ndim, Dtype type, Device device);
```
**Example:**
```c
float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
int64_t shape[] = {2, 2};
Tensor *t = tensor_from_arr(data, shape, 2, FLOAT32, CPU);
tensor_print(t);
// Output:
// Tensor of shape (2,2):
// [1.00,2.00],
// [3.00,4.00]
```

---

### 2. Tensor Operations

Perform mathematical or manipulation operations on tensors.

#### Mathematical Operations

-   `tensor_add(a, b)`: Element-wise addition.
-   `tensor_sub(a, b)`: Element-wise subtraction.
-   `tensor_mul(a, b)`: Element-wise multiplication.
-   `tensor_div(a, b)`: Element-wise division.
-   `tensor_matmul(a, b)`: Matrix multiplication with support for batch dimensions.
-   `tensor_sum(a)`: Computes the sum of all elements, returning a scalar tensor.

**Example:**
```c
int64_t shape[] = {2, 2};
float val_a = 2.0f;
Tensor *a = tensor_full(shape, 2, FLOAT32, CPU, &val_a);

float val_b = 3.0f;
Tensor *b = tensor_full(shape, 2, FLOAT32, CPU, &val_b);

// Element-wise multiplication
Tensor *c = tensor_mul(a, b);
tensor_print(c);
// Output: [6.00,6.00],[6.00,6.00]

// Matrix multiplication
Tensor *d = tensor_matmul(a, b);
tensor_print(d);
// Output: [12.00,12.00],[12.00,12.00]
```

#### Manipulation Operations

-   `tensor_reshape(a, new_shape, new_ndim)`: Returns a tensor with a new shape. May copy data if the original tensor is not contiguous.
-   `tensor_transpose(a, dim0, dim1)`: Returns a new tensor with two dimensions swapped.
-   `tensor_t(a)`: Transposes a 1D or 2D tensor. A convenient wrapper around `tensor_transpose`.

---

### 3. Autograd Engine

The autograd engine tracks operations to compute gradients automatically. This is the magic behind training neural networks!

#### `tensor_set_require_grad`

Enables gradient tracking for a tensor. Call this on any tensor (like weights or biases) that you want to optimize.
```c
void tensor_set_require_grad(Tensor *a, int requires_grad); // 1 for true, 0 for false
```

#### `tensor_backward`

Computes the gradient of a tensor with respect to all leaf tensors that have `requires_grad=1`. It traverses the computation graph backwards from the calling tensor.

**Note:** When `tensor_backward` is called on a non-scalar tensor, it computes the gradient of the **sum** of that tensor's elements.

```c
void tensor_backward(Tensor *a, Tensor *prev_grad); // Pass NULL to start with a gradient of ones
```

**Example:**
Let `y = a * b`. We want to find the gradient of `sum(y)` with respect to `a` and `b`.
```c
// Create tensor 'a'
float data_a[] = {1.0, 2.0, 3.0, 4.0};
int64_t shape[] = {2, 2};
Tensor *a = tensor_from_arr(data_a, shape, 2, FLOAT32, CPU);
tensor_set_require_grad(a, 1); // Track gradients for a

// Create tensor 'b'
float data_b[] = {5.0, 6.0, 7.0, 8.0};
Tensor *b = tensor_from_arr(data_b, shape, 2, FLOAT32, CPU);
tensor_set_require_grad(b, 1); // Track gradients for b

// y = a * b (element-wise)
Tensor *y = tensor_mul(a, b);

// Compute gradients of sum(y) w.r.t. the leaf tensors (a and b).
tensor_backward(y, NULL);

// Print gradients. The gradient of `sum(a*b)` w.r.t. `a` is `b`.
printf("Gradient of a (should be equal to b):\n");
tensor_print(a->grad);

// The gradient of `sum(a*b)` w.r.t. `b` is `a`.
printf("\nGradient of b (should be equal to a):\n");
tensor_print(b->grad);
```

---

### 4. Neural Network (nn) Module

The `nn` module provides high-level abstractions for building networks.

#### `module_constructor`

Creates a `Module` object, which acts as a container for all trainable parameters (weights and biases) in a model.
```c
Module *module_constructor();
```

#### `Linear`

Applies a linear transformation: `y = x @ W + b`. Weights and biases are automatically created, initialized, and registered as parameters in the module.
```c
Tensor *Linear(Module *m, Tensor *x, int64_t out_features, int bias, int layer, Allocated_tensors *Al);
```

-   `m`: Pointer to the `Module` that will store the parameters.
-   `x`: The input tensor of shape `(..., in_features)`.
-   `out_features`: The number of output features for the layer.
-   `bias`: Flag (1 for true, 0 for false) to include a trainable bias term.
-   `layer`: The index of the layer, used to retrieve or create the correct parameters from the module.
-   `Al`: A pointer to a struct that tracks intermediate tensors for memory management.

#### `tensor_relu`

Applies the Rectified Linear Unit activation function element-wise: `ReLU(x) = max(0, x)`.
```c
Tensor *tensor_relu(Tensor *x);
```

#### `optimizer_step`

Updates the module's parameters using their computed gradients: `param = param - lr * param.grad`.
```c
void optimizer_step(Module *module, float lr);
```

#### `module_zero_grad`

Resets the gradients of all parameters in a module to zero. This must be called at the start of each training iteration.
```c
void module_zero_grad(Module *module);
```

#### `module_free`

Frees the module and all its associated parameters.
```c
void module_free(Module *module);
```

---

## ⚡ Putting It All Together: A Complete Example

The `main.c` file demonstrates how to use these components to build and train a simple neural network.

The network has the following architecture:
1.  Linear Layer (2 -> 2) + Bias
2.  ReLU Activation
3.  Linear Layer (2 -> 2) + Bias
4.  ReLU Activation
5.  Linear Layer (2 -> 1) + Bias

The training loop performs the standard steps:

1.  **Forward Pass**: Get predictions from the model.
2.  **Loss Calculation**: Compute the error.
3.  **Backward Pass**: Compute gradients for all parameters.
4.  **Optimizer Step**: Update parameters to reduce the error.
5.  **Zero Gradients**: Reset for the next iteration.

```c
#include "headers/tensor.h"
#include "headers/nn.h"

int main()
{
    // Set seed for reproducible random weight initialization
    tensor_set_seed(1337);

    // 1. Prepare the Dataset
    float data[20][2] = {
        {0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}, {0.7, 0.8}, {0.2, 0.3},
        {0.4, 0.6}, {0.6, 0.8}, {0.8, 0.9}, {0.1, 0.4}, {0.3, 0.5},
        {0.5, 0.7}, {0.7, 0.9}, {0.2, 0.5}, {0.4, 0.7}, {0.6, 0.9},
        {0.1, 0.3}, {0.3, 0.6}, {0.5, 0.8}, {0.7, 0.7}, {0.9, 0.8}
    };
    float labels[20] = {
        0.17, 0.37, 0.57, 0.77, 0.27, 0.53, 0.73, 0.87, 0.30, 0.43,
        0.63, 0.83, 0.40, 0.60, 0.80, 0.23, 0.50, 0.70, 0.70, 0.83
    };
    int64_t data_shape[] = {1, 20, 2};
    int64_t label_shape[] = {20, 1};
    // Create tensors using [tensor_from_arr](#tensor_from_arr)
    Tensor *data_t = tensor_from_arr(data, data_shape, 3, FLOAT32, CPU);
    Tensor *label_t = tensor_from_arr(labels, label_shape, 2, FLOAT32, CPU);

    // 2. Initialize Model, Optimizer, and Memory Tracker
    // Create a module to hold parameters with [module_constructor](#module_constructor)
    Module *module = module_constructor();
    Allocated_tensors At; // For tracking intermediate tensors inside Linear layers
    At.ptrs = NULL;
    int epoch = 2000;
    float lr = 0.001;

    // --- 3. Training Loop ---
    for (int i = 0; i <= epoch; i++)
    {
        // === FORWARD PASS ===
        Tensor *l1 = Linear(module, data_t, 2, 1, 0, &At);
        Tensor *r1 = tensor_relu(l1);
        Tensor *l2 = Linear(module, r1, 2, 1, 1, &At);
        Tensor *r2 = tensor_relu(l2);
        Tensor *pred = Linear(module, r2, 1, 1, 2, &At);
        if (i == 0)
        {
            printf("first prediction before training: \n");
            tensor_print(pred);
        }
        // === LOSS CALCULATION (Sum of Squared Errors) ===
        Tensor *sub = tensor_sub(label_t, pred);
        Tensor *pow = tensor_mul(sub, sub);
        Tensor *loss = tensor_sum(pow);

        if (i % 500 == 0 || i == epoch) {
            printf("--- Iteration: %d, Loss: %.4f ---\n", i, ((float*)loss->data)[0]);
        }

        // === BACKPROPAGATION ===
        // Compute gradients from the loss using [tensor_backward](#tensor_backward)
        tensor_backward(loss, NULL);

        // === OPTIMIZER STEP ===
        // Update weights and biases with the [optimizer_step](#optimizer_step)
        optimizer_step(module, lr);

        // === ZERO GRADIENTS ===
        // Reset gradients for the next iteration with [module_zero_grad](#module_zero_grad)
        module_zero_grad(module);

        // === MEMORY CLEANUP for this iteration ===
        tensor_free(loss);
        tensor_free(pow);
        tensor_free(sub);
        tensor_free(pred);
        tensor_free(r2);
        tensor_free(l2);
        tensor_free(r1);
        tensor_free(l1);
    }
    
    printf("\n--- Training Finished ---\n");
    // Get final predictions to compare against labels
    Tensor *final_preds = Linear(module, data_t, 2, 1, 0, &At);
    final_preds = tensor_relu(final_preds);
    final_preds = Linear(module, final_preds, 2, 1, 1, &At);
    final_preds = tensor_relu(final_preds);
    final_preds = Linear(module, final_preds, 1, 1, 2, &At);

    printf("Final Predictions:\n");
    tensor_print(final_preds);
    printf("\nTrue Labels:\n");
    tensor_print(label_t);


    // --- 4. Final Cleanup ---
    // Free the module and all its parameters with [module_free](#module_free)
    free_allocated_tensors(&At);
    tensor_free(label_t);
    tensor_free(data_t);
    module_free(module);

    return 0;

}
```
