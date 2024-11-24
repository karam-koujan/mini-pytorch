#ifndef TENSOR_H
#define TENSOR_H

#include <stdarg.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum Dtype
{
	FLOAT32,
	DOUBLE,
	INT32,
	INT64
};

enum Device
{
	CPU,
	GPU
};

typedef	struct
{
	void *data;
	int *shape;
	int *strides;
	enum Dtype dtype;
	enum Device device;
	void *grad;
	int	requires_grad;
	int num_dims;
} Tensor;

#endif