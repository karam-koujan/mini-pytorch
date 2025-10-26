# Mini-PyTorch

Mini-PyTorch is a small, educational tensor library written in pure C, inspired by the core functionalities of PyTorch. It provides a dynamic Tensor object, an automatic differentiation engine (autograd), and basic building blocks for creating neural networks.

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

# Run the executable
./mini_pytorch
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
Tensor *t = tensor_zeros(shape, 2, DOUBLE, CPU);
tensor_print(t);
// Output:
// Tensor of shape (2,3):
// [[0.00,0.00,0.00],
// [0.00,0.00,0.00]]
```

#### `tensor_ones`
Creates a tensor of a given shape filled with ones.
```c
Tensor *tensor_ones(const int64_t *shape, int64_t ndim, Dtype type, Device device);
```
**Example:**
```c
int64_t shape[] = {2, 3};
Tensor *t = tensor_ones(shape, 2, DOUBLE, CPU);
tensor_print(t);
// Output:
// Tensor of shape (2,3):
// [[1.00,1.00,1.00],
// [1.00,1.00,1.00]]
```

#### `tensor_full`
Creates a tensor of a given shape filled with a specified scalar value.
```c
Tensor *tensor_full(const int64_t *shape, int64_t ndim, Dtype type, Device device, void *val);
```
**Example:**
```c
int64_t shape[] = {2, 3};
double val = 7.5;
Tensor *t = tensor_full(shape, 2, DOUBLE, CPU, &val);
tensor_print(t);
// Output:
// Tensor of shape (2,3):
// [[7.50,7.50,7.50],
// [7.50,7.50,7.50]]
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
Tensor *t = tensor_urand(shape, 2, DOUBLE, CPU, -10.0, 10.0);
tensor_print(t);
```

#### `tensor_from_arr`
Creates a tensor from an existing C array.
```c
Tensor *tensor_from_arr(void *arr, const int64_t *shape, int64_t ndim, Dtype type, Device device);
```
**Example:**
```c
double data[] = {1.0, 2.0, 3.0, 4.0};
int64_t shape[] = {2, 2};
Tensor *t = tensor_from_arr(data, shape, 2, DOUBLE, CPU);
tensor_print(t);
// Output:
// Tensor of shape (2,2):
// [[1.00,2.00],
// [3.00,4.00]]
```

### 2. Tensor Operations

#### Mathematical Operations
These functions perform element-wise or matrix operations.

`tensor_add(a, b)`: Element-wise addition.
`tensor_sub(a, b)`: Element-wise subtraction.
`tensor_mul(a, b)`: Element-wise multiplication.
`tensor_div(a, b)`: Element-wise division.
`tensor_mm(a, b)`: Matrix multiplication for 2D tensors.
`tensor_matmul(a, b)`: Matrix multiplication with support for broadcasting batch dimensions.

**Example (Math Ops):**
```c
int64_t shape[] = {2, 2};
double val_a = 2.0;
Tensor *a = tensor_full(shape, 2, DOUBLE, CPU, &val_a);

double val_b = 3.0;
Tensor *b = tensor_full(shape, 2, DOUBLE, CPU, &val_b);

// Element-wise multiplication
Tensor *c = tensor_mul(a, b);
tensor_print(c);
// Output:
// Tensor of shape (2,2):
// [[6.00,6.00],
// [6.00,6.00]]

// Matrix multiplication
Tensor *d = tensor_mm(a, b);
tensor_print(d);
// Output:
// Tensor of shape (2,2):
// [[12.00,12.00],
// [12.00,12.00]]
```

#### Manipulation Operations
These functions change the shape or layout of a tensor.

`tensor_reshape(a, new_shape, new_ndim)`: Returns a tensor with a new shape. May copy data if the original tensor is not contiguous.
`tensor_transpose(a, dim0, dim1)`: Swaps two dimensions of a tensor.
`tensor_t(a)`: Transposes a 1D or 2D tensor.
`tensor_permute(a, dims, num_dims)`: Permutes the dimensions of a tensor according to a specified order.

**Example (Transpose):**
```c
double data[] = {1, 2, 3, 4};
int64_t shape[] = {2, 2};
Tensor *a = tensor_from_arr(data, shape, 2, DOUBLE, CPU);
printf("Original Tensor:\n");
tensor_print(a);

Tensor *a_t = tensor_transpose(a, 0, 1);
printf("Transposed Tensor:\n");
tensor_print(a_t);
// Original Output:
// [[1.00,2.00],
// [3.00,4.00]]
// Transposed Output:
// [[1.00,3.00],
// [2.00,4.00]]
```

#### Utility Functions
`tensor_print(t)`: Prints a formatted representation of the tensor.
`tensor_infos(t)`: Prints detailed metadata about the tensor (shape, strides, dtype, etc.).


### 3. Autograd Engine

The autograd engine tracks operations to compute gradients automatically.

#### `tensor_set_require_grad`
Enables or disables gradient tracking for a tensor. This should be called on leaf tensors (tensors you create directly) that you want to compute gradients with respect to.

```c
void tensor_set_require_grad(Tensor *a, int requires_grad); // 1 for true, 0 for false
```

#### `tensor_backward`
Computes the gradient of a tensor with respect to all leaf tensors that have `requires_grad=1`.

**Important:** When calling `tensor_backward` on a tensor with multiple elements (a non-scalar), you are implicitly asking for the gradient of the **sum** of its elements. The function automatically creates an initial gradient tensor of ones with the same shape as the output tensor.

```c
void tensor_backward(Tensor *a, Tensor *prev_grad); // Pass NULL to use the default gradient of ones
```

**Example (Autograd with Multi-Dimensional Tensors):**

Let's compute the gradient for an element-wise multiplication of two 2x2 tensors.
If `y = a * b`, then the loss `L` is implicitly `sum(y)`. The gradient `dL/da` will be `b`, and `dL/db` will be `a`.

```c
// Let y = a * b, where a and b are 2x2 matrices.
// We want to compute the gradients of the sum of y's elements
// with respect to a and b.

// Create tensor 'a'
double data_a[] = {1.0, 2.0, 3.0, 4.0};
int64_t shape[] = {2, 2};
Tensor *a = tensor_from_arr(data_a, shape, 2, DOUBLE, CPU);
tensor_set_require_grad(a, 1); // Track gradients for a

// Create tensor 'b'
double data_b[] = {5.0, 6.0, 7.0, 8.0};
Tensor *b = tensor_from_arr(data_b, shape, 2, DOUBLE, CPU);
tensor_set_require_grad(b, 1); // Track gradients for b

// y = a * b (element-wise)
// y will be [[5.0, 12.0], [21.0, 32.0]]
Tensor *y = tensor_mul(a, b);

// Compute gradients. Since y is not a scalar, this computes the gradient
// of sum(y) w.r.t. the leaf tensors. It's equivalent to providing
// an initial gradient of ones.
tensor_backward(y, NULL);

// Print gradients
// The gradient of `sum(a*b)` w.r.t. `a` is `b`.
printf("Original tensor b:\n");
tensor_print(b);
printf("\nGradient of a (should be equal to b):\n");
tensor_print(a->grad);

// The gradient of `sum(a*b)` w.r.t. `b` is `a`.
printf("\nOriginal tensor a:\n");
tensor_print(a);
printf("\nGradient of b (should be equal to a):\n");
tensor_print(b->grad);

/*
Expected Output:

Original tensor b:
Tensor of shape (2,2):
[[5.00,6.00],
[7.00,8.00]]

Gradient of a (should be equal to b):
Tensor of shape (2,2):
[[5.00,6.00],
[7.00,8.00]]

Original tensor a:
Tensor of shape (2,2):
[[1.00,2.00],
[3.00,4.00]]

Gradient of b (should be equal to a):
Tensor of shape (2,2):
[[1.00,2.00],
[3.00,4.00]]
*/
```

### 4. Neural Network Module (`nn`)

The `nn` module provides building blocks for creating neural networks.

#### `module_constructor`
Creates a `Module` object, which acts as a container for all trainable parameters (weights and biases) in a model.

```c
Module *module_constructor();
```

#### `Linear`
Applies a linear transformation to the input data: `y = xA^T + b`. The weights and biases are automatically created, initialized, and registered as parameters in the provided module.

```c
Tensor *Linear(Module *m, Tensor *x, int64_t out_features, int bias);
```
- `m`: A pointer to the `Module` that will store the parameters.
- `x`: The input tensor of shape `(..., in_features)`.
- `out_features`: The number of output features for the layer.
- `bias`: An integer flag (1 for true, 0 for false) to include a trainable bias term.

#### `relu`
Applies the Rectified Linear Unit activation function element-wise. `ReLU(x) = max(0, x)`. This is an in-place operation.

```c
Tensor *relu(Tensor *x);
```

#### `parameters_print`
Prints all parameters registered within a `Module`.

```c
void parameters_print(Tensor **parameters);
```

**Example (Building a Simple Network):**
```c
// 1. Create a module to hold model parameters
Module *model = module_constructor();

// 2. Define an input tensor (e.g., batch of 1, 10 features)
int64_t input_shape[] = {1, 10};
Tensor *x = tensor_rand(input_shape, 2, DOUBLE, CPU);

// 3. Forward pass through a simple network
// Layer 1: 10 input features -> 32 output features
Tensor *hidden = Linear(model, x, 32, 1); // `1` enables bias
// Activation function
relu(hidden);
// Layer 2: 32 input features -> 5 output features
Tensor *output = Linear(model, hidden, 5, 1);

// 4. Print results
printf("Input Tensor:\n");
tensor_print(x);
printf("\nOutput Tensor:\n");
tensor_print(output);
printf("\nModel Parameters:\n");
parameters_print(model->parameters);

// 5. Example backward pass (assuming `output` is a scalar loss)
// For demonstration, let's create a dummy scalar from the output
Tensor *loss = tensor_add(output, output); // Dummy operation to keep graph
tensor_backward(loss, NULL);

printf("\n---Gradients after backward pass---\n");
parameters_print(model->parameters); // The .grad fields will now be populated
```