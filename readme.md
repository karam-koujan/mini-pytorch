Excellent idea. Adding internal links will significantly improve the navigation and usability of the documentation.

Here is the fully updated `README.md` with a clickable table of contents and cross-references within the text.

---

# Mini-PyTorch

Mini-PyTorch is a small, educational tensor library written in pure C, inspired by the core functionalities of PyTorch. It provides a dynamic Tensor object, an automatic differentiation engine (autograd), and basic building blocks for creating neural networks.

This project is intended for educational purposes to understand the inner workings of a deep learning framework.

## Table of Contents

-   [Features](#features)
-   [How to Build](#how-to-build)
-   [API Documentation & Examples](#api-documentation--examples)
    -   [1. Tensor Creation](#1-tensor-creation)
        -   [`tensor_zeros`](#tensor_zeros)
        -   [`tensor_ones`](#tensor_ones)
        -   [`tensor_full`](#tensor_full)
        -   [`tensor_rand`](#tensor_rand)
        -   [`tensor_urand`](#tensor_urand)
        -   [`tensor_from_arr`](#tensor_from_arr)
    -   [2. Tensor Operations](#2-tensor-operations)
        -   [Mathematical Operations](#mathematical-operations)
        -   [Manipulation Operations](#manipulation-operations)
        -   [Utility Functions](#utility-functions)
    -   [3. Autograd Engine](#3-autograd-engine)
        -   [`tensor_set_require_grad`](#tensor_set_require_grad)
        -   [`tensor_backward`](#tensor_backward)
    -   [4. Neural Network Module (nn)](#4-neural-network-module-nn)
        -   [`module_constructor`](#module_constructor)
        -   [`Linear`](#linear)
        -   [`tensor_relu`](#tensor_relu)
        -   [`mse`](#mse)
        -   [`optimizer_step`](#optimizer_step)
        -   [`module_zero_grad`](#module_zero_grad)
        -   [`parameters_print`](#parameters_print)
        -   [`module_free`](#module_free)
-   [Putting It All Together: A Complete Example](#putting-it-all-together-a-complete-example)
-   [Known Issues](#known-issues)

## Features

-   **Tensor Operations**: Create and manipulate multi-dimensional arrays (tensors).
-   **Broadcasting**: Automatic expansion of tensor shapes for element-wise operations.
-   **Automatic Differentiation (Autograd)**: Automatically compute gradients for any sequence of tensor operations.
-   **Neural Network Primitives**: Basic layers like `Linear` and activation functions like `ReLU` to build simple models.

## How to Build

To build the library and run the main executable, use the provided `Makefile`.

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

## API Documentation & Examples

### 1. Tensor Creation

These functions are used to create new tensors.

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

#### `tensor_rand`
Creates a tensor of a given shape with random values uniformly distributed between 0 and 1.
```c
Tensor *tensor_rand(const int64_t *shape, int64_t ndim, Dtype type, Device device);
```

#### `tensor_urand`
Creates a tensor of a given shape with random values uniformly distributed between a specified `min` and `max`.
```c
Tensor *tensor_urand(const int64_t *shape, int64_t ndim, Dtype type, Device device, double min, double max);
```
**Example:**
```c
tensor_set_seed(1337); // for reproducibility
int64_t shape[] = {2, 2};
Tensor *t = tensor_urand(shape, 2, FLOAT32, CPU, -10.0, 10.0);
tensor_print(t);
```

#### `tensor_from_arr`
Creates a tensor from an existing C array.
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

### 2. Tensor Operations

#### Mathematical Operations
These functions perform element-wise or matrix operations.

-   `tensor_add(a, b)`: Element-wise addition.
-   `tensor_sub(a, b)`: Element-wise subtraction.
-   `tensor_mul(a, b)`: Element-wise multiplication.
-   `tensor_div(a, b)`: Element-wise division.
-   `tensor_mm(a, b)`: Matrix multiplication for 2D tensors.
-   `tensor_matmul(a, b)`: Matrix multiplication with support for broadcasting batch dimensions.
-   `tensor_sum(a)`: Computes the sum of all elements in the tensor, returning a scalar tensor.
-   `tensor_mean(a)`: Computes the mean of all elements in the tensor, returning a scalar tensor.

**Example (Math Ops):**
```c
int64_t shape[] = {2, 2};
float val_a = 2.0f;
Tensor *a = tensor_full(shape, 2, FLOAT32, CPU, &val_a);

float val_b = 3.0f;
Tensor *b = tensor_full(shape, 2, FLOAT32, CPU, &val_b);

// Element-wise multiplication
Tensor *c = tensor_mul(a, b);
tensor_print(c);
// Output:
// Tensor of shape (2,2):
// [6.00,6.00],
// [6.00,6.00]

// Matrix multiplication
Tensor *d = tensor_mm(a, b);
tensor_print(d);
// Output:
// Tensor of shape (2,2):
// [12.00,12.00],
// [12.00,12.00]
```

#### Manipulation Operations
These functions change the shape or layout of a tensor.

-   `tensor_reshape(a, new_shape, new_ndim)`: Returns a tensor with a new shape. May copy data if the original tensor is not contiguous.
-   `tensor_transpose(a, dim0, dim1)`: Swaps two dimensions of a tensor by creating a deep copy.
-   `tensor_t(a)`: Transposes a 1D or 2D tensor.
-   `tensor_permute(a, dims, num_dims)`: Permutes the dimensions of a tensor according to a specified order (in-place).

**Example (Transpose):**
```c
float data[] = {1, 2, 3, 4};
int64_t shape[] = {2, 2};
Tensor *a = tensor_from_arr(data, shape, 2, FLOAT32, CPU);
printf("Original Tensor:\n");
tensor_print(a);

Tensor *a_t = tensor_t(a);
printf("Transposed Tensor:\n");
tensor_print(a_t);
// Original Output:
// Tensor of shape (2,2):
// [1.00,2.00],
// [3.00,4.00]
// Transposed Output:
// Tensor of shape (2,2):
// [1.00,3.00],
// [2.00,4.00]
```

#### Utility Functions
-   `tensor_print(t)`: Prints a formatted representation of the tensor.
-   `tensor_infos(t)`: Prints detailed metadata about the tensor (shape, strides, dtype, etc.).

### 3. Autograd Engine

The autograd engine tracks operations to compute gradients automatically.

#### `tensor_set_require_grad`
Enables or disables gradient tracking for a tensor. This should be called on leaf tensors (tensors you create directly) that you want to compute gradients with respect to.

```c
void tensor_set_require_grad(Tensor *a, int requires_grad); // 1 for true, 0 for false
```

#### `tensor_backward`
Computes the gradient of a tensor with respect to all leaf tensors that have `requires_grad=1`. It traverses the computation graph backwards from the calling tensor.

**Important:** When calling `tensor_backward` on a non-scalar tensor, you are implicitly asking for the gradient of the **sum** of its elements. The function automatically creates an initial gradient of ones to start the backpropagation process.

```c
void tensor_backward(Tensor *a, Tensor *prev_grad); // Pass NULL to use the default gradient of ones
```

**Example:**
```c
// Let y = a * b, where a and b are 2x2 matrices.
// We want to compute the gradients of the sum of y's elements
// with respect to a and b.

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

// Compute gradients. Since y is not a scalar, this computes the gradient
// of sum(y) w.r.t. the leaf tensors.
tensor_backward(y, NULL);

// Print gradients
// The gradient of `sum(a*b)` w.r.t. `a` is `b`.
printf("Gradient of a (should be equal to b):\n");
tensor_print(a->grad);

// The gradient of `sum(a*b)` w.r.t. `b` is `a`.
printf("\nGradient of b (should be equal to a):\n");
tensor_print(b->grad);
```

### 4. Neural Network Module (nn)

The `nn` module provides building blocks for creating neural networks.

#### `module_constructor`
Creates a `Module` object, which acts as a container for all trainable parameters (weights and biases) in a model.
```c
Module *module_constructor();
```

#### `Linear`
Applies a linear transformation to the input data: `y = x @ W + b`. The weights and biases are automatically created, initialized using Xavier/Glorot initialization, and registered as parameters in the provided module.
```c
Tensor *Linear(Module *m, Tensor *x, int64_t out_features, int bias, int layer);
```
-   `m`: A pointer to the `Module` that will store the parameters.
-   `x`: The input tensor of shape `(..., in_features)`.
-   `out_features`: The number of output features for the layer.
-   `bias`: An integer flag (1 for true, 0 for false) to include a trainable bias term.
-   `layer`: The index of the layer, used to retrieve or create the correct parameters from the module.

#### `tensor_relu`
Applies the Rectified Linear Unit activation function element-wise: `ReLU(x) = max(0, x)`.
```c
Tensor *tensor_relu(Tensor *x);
```

#### `mse`
Computes the Mean Squared Error between two tensors. Note: the current implementation calculates the Sum of Squared Errors: `sum((y - y_pred)^2)`.
```c
Tensor *mse(Tensor *y, Tensor *y_pred);
```

#### `optimizer_step`
Updates the module's parameters using their computed gradients. It performs the update: `param = param - lr * param.grad`.
```c
void optimizer_step(Module *module, float lr);
```

#### `module_zero_grad`
Resets the gradients of all parameters in a module to zero. This should be called at the start of each training iteration.
```c
void module_zero_grad(Module *module);
```

#### `parameters_print`
Prints all parameters (weights and biases) registered within a `Module`.
```c
void parameters_print(Tensor **parameters);
```

#### `module_free`
Frees the module and all its associated parameters.
```c
void module_free(Module *module);
```

## Putting It All Together: A Complete Example

The `main.c` file demonstrates how to use these components to build and train a simple neural network.

The network has the following architecture:
1.  Linear Layer (2 input features, 2 output features) + Bias
2.  ReLU Activation
3.  Linear Layer (2 input features, 2 output features) + Bias
4.  ReLU Activation
5.  Linear Layer (2 input features, 1 output feature) + Bias

The training loop performs the following steps:
1.  **Forward Pass**: Data is passed through the network to get a prediction.
2.  **Loss Calculation**: The Sum of Squared Errors is calculated between the prediction and the true labels.
3.  **Backward Pass**: `tensor_backward()` is called on the loss to compute gradients for all model parameters.
4.  **Optimizer Step**: The model's parameters are updated using the gradients.
5.  **Zero Gradients**: The gradients are reset for the next iteration.

```c
#include "headers/tensor.h"
#include "headers/print.h"
#include "headers/nn.h"
#include <time.h>

// Helper to print logs during training
void print_logs(Tensor *pred, Tensor *label_t, Tensor *loss, int epoch, int i)
{
    if (i % 500 == 0 || i == epoch || i == 0)
    {
        printf("--- Iteration: %i ---\n", i);
        printf("True label (first 5):\n");
        // Simplified print for brevity in README
        for(int k=0; k<5; ++k) printf("%.2f ", ((float*)label_t->data)[k]);
        printf("\n");

        printf("Predicted label (first 5):\n");
        for(int k=0; k<5; ++k) printf("%.2f ", ((float*)pred->data)[k]);
        printf("\n");
        
        printf("Loss value:\n");
        tensor_print(loss);
        printf("\n");
    }
}

int main()
{
    // Set seed for reproducible random weight initialization
    tensor_set_seed(1337);

    // Sample dataset and labels
    float data[20][2] = {
        {0.1, 0.2}, /* ... more data ... */ {0.9, 0.8}
    };
    float labels[20] = {
        0.17, /* ... more labels ... */ 0.83
    };

    // Create tensors from C arrays using [tensor_from_arr](#tensor_from_arr)
    int64_t data_shape[] = {1, 20, 2};
    int64_t label_shape[] = {20, 1};
    Tensor *data_t = tensor_from_arr(data, data_shape, 3, FLOAT32, CPU);
    Tensor *label_t = tensor_from_arr(labels, label_shape, 2, FLOAT32, CPU);

    // Create a module to hold the network parameters with [module_constructor](#module_constructor)
    Module *module = module_constructor();
    int epoch = 2000;
    float lr = 0.001;

    // --- Training Loop ---
    for (int i = 0; i <= epoch; i++)
    {
        // 1. Forward pass
        Tensor *l1 = Linear(module, data_t, 2, 1, 0);
        Tensor *r1 = tensor_relu(l1);
        Tensor *l2 = Linear(module, r1, 2, 1, 1);
        Tensor *r2 = tensor_relu(l2);
        Tensor *pred = Linear(module, r2, 1, 1, 2);

        // 2. Loss calculation (Sum of Squared Errors)
        Tensor *sub = tensor_sub(label_t, pred);
        Tensor *pow = tensor_mul(sub, sub);
        Tensor *loss = tensor_sum(pow);

        print_logs(pred, label_t, loss, epoch, i);

        // 3. Backpropagation: compute gradients from the loss using [tensor_backward](#tensor_backward)
        tensor_backward(loss, NULL);

        // 4. Update weights and biases with the [optimizer_step](#optimizer_step)
        optimizer_step(module, lr);

        // 5. Zero out gradients for the next iteration with [module_zero_grad](#module_zero_grad)
        module_zero_grad(module);

        // Free intermediate tensors from the forward pass and loss calculation
        tensor_free(l1);
        tensor_free(r1);
        tensor_free(l2);
        tensor_free(r2);
        tensor_free(pow);
        tensor_free(sub);
        tensor_free(pred);
        tensor_free(loss);
    }

    // Clean up all remaining resources
    tensor_free(label_t);
    tensor_free(data_t);
    module_free(module);

    return 0;
}
```

## Known Issues

⚠️ **Memory Management**: This project currently has known memory leaks. The tensor and module freeing logic is not complete, and running complex models or long training loops will result in significant memory consumption. This is a key area that needs to be addressed for the library to be more robust.