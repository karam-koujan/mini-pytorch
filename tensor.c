#include "tensor.h"



int	*create_shape(va_list arg,int dim)
{
	int *shape = (int *)malloc(dim * sizeof(int));
	if (!shape)
		return NULL;
	for(int i = 0; i < dim; i++)
	{
		shape[i] = va_arg(arg, int);
	}
	return shape;
}

int	*create_stride(int num_dims, int *shape)
{
	int *stride = malloc(num_dims);
	if (!stride)
		return NULL;
	int sum = 1;
	for(int i = num_dims - 1; i >= 0; i--)
	{
		stride[i] = sum;
		sum *= shape[i];
	}
	return stride;
}

void	add_options(va_list arg,Tensor *tensor)
{
	tensor->device = CPU;
	tensor->dtype = FLOAT32;
	char	*device = va_arg(arg,char *);
	char	*dtype = va_arg(arg,char *);

	if (device && !strcmp(device,"gpu"))
	{
		tensor->device = GPU;
	}
	if (dtype && !strcmp(dtype,"int"))
	{
		tensor->dtype = INT32;
	}

}
float	*create_empty_data(int dim,int *shape)
{
	int data_size = 1;
	for(int i = 0; i < dim; i++)
	{
		data_size*=shape[i];
	}
	float	*data = malloc(data_size * sizeof(float));
	if (!data)
		return NULL;
	return (data);
}
int		tensor_entries_len(Tensor *tensor)
{
	int count = 1;
	for(int i = 0; i < tensor->num_dims; i++)
	{
		count*=tensor->shape[i];
	}
	return count;
}

// void	tensor_fill(Tensor *tensor, float num)
// {

// 	for(int i = 0; i < dim)
// }
Tensor	 tensor_empty(int dim,...)
{
	va_list arg;
	Tensor tensor;

	va_start(arg,dim);
	tensor.shape =  create_shape(arg,dim);
	tensor.num_dims = dim;
	tensor.strides = create_stride(tensor.num_dims, tensor.shape);
	add_options(arg,&tensor);
	tensor.data = (float *)create_empty_data(dim,tensor.shape);
	va_end(arg);
	return tensor;
}

Tensor tensor_zeros(int dim,...)
{
	va_list arg;
	Tensor tensor;

	va_start(arg,dim);
	tensor.shape =  create_shape(arg,dim);
	tensor.num_dims = dim;
	tensor.strides = create_stride(tensor.num_dims, tensor.shape);
	add_options(arg,&tensor);
	tensor.data = (float *)create_empty_data(dim,tensor.shape);
	va_end(arg);
	return tensor;
}

int main()
{
	Tensor tensor = tensor_empty(3,5,2,2,NULL,NULL);
	printf("%d",tensor_entries_len(&tensor));
}