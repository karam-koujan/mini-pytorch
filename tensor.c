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

Tensor	create_empty_tensor(int dim,...)
{
	va_list arg;
	Tensor tensor;

	va_start(arg,dim);
	tensor.shape =  create_shape(arg,dim);
	va_end(arg);
	return tensor;
}
