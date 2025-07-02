#ifndef TENSOR_H
#define TENSOR_H

#include <stdarg.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

typedef enum e_type
{
	FLOAT32,
	DOUBLE,
	INT32,
	INT64
} Dtype;

typedef enum e_device
{
	CPU,
	GPU
} Device;


typedef	struct
{
	void *data;
	int64_t *shape;
	int64_t *strides;
	Dtype dtype;
	Device device;
	void *grad;
	int size;
	int	requires_grad;
	int num_dims;
	int	is_leaf;
	void *grad_fn;
} Tensor;

typedef struct Node
{
	Tensor *grad;
	Tensor **saved_tensors;
	Tensor **(*calculate_gradient)(struct Node *node,Tensor *grad);
}	Grad_Node;

typedef struct
{
	Tensor **parameters;
} Module;

#endif