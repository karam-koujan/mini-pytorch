#include "tensor.h"





int	*create_shape(va_list arg,int dim)
{
	int *shape = (int *)malloc(dim * sizeof(int));
	for(int i = 0; i < dim; i++)
	{
		shape[i] = va_arg(arg, int);
	}
	return shape;
}

int	*create_stride(int num_dims, int *shape)
{
	int *stride = malloc(num_dims);
	int sum = 1;
	for(int i = num_dims - 1; i >= 0; i--)
	{
		stride[i] = sum;
		sum *= shape[i];
	}
	return (stride);
}

Tensor	create_empty_tensor(int dim,...)
{
	va_list arg;
	Tensor tensor;

	va_start(arg,dim);
	tensor.shape =  create_shape(arg,dim);
	tensor.num_dims = dim;
	tensor.strides = create_stride(tensor.num_dims, tensor.shape);
	va_end(arg);
	return tensor;
}

