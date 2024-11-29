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

void	tensor_fill(Tensor *tensor, float num)
{
	int tensor_entries_num = tensor_entries_len(tensor);
	float *data = (float *)tensor->data;
	for(int i = 0; i < tensor_entries_num;i++)
	{
		data[i] = num;
	}
}

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
	tensor_fill(&tensor,0);
	va_end(arg);
	return tensor;
}



Tensor tensor_ones(int dim,...)
{
	va_list arg;
	Tensor tensor;

	va_start(arg,dim);
	tensor.shape =  create_shape(arg,dim);
	tensor.num_dims = dim;
	tensor.strides = create_stride(tensor.num_dims, tensor.shape);
	add_options(arg,&tensor);
	tensor.data = (float *)create_empty_data(dim,tensor.shape);
	tensor_fill(&tensor,1);
	va_end(arg);
	return tensor;
}
void print_tensor_recursive(float *data, int *shape, int num_dims, int index, int depth) {
    if (depth == num_dims) {
        printf("%f", data[index]);
    } else {
        for (int i = 0; i < shape[depth]; i++) {
            if (depth < num_dims - 1) {
                printf("[");
            }
            print_tensor_recursive(data, shape, num_dims, index + i * (index == 0 ? 1 : shape[depth - 1]), depth + 1);
            if (depth < num_dims - 1) {
                printf("]\n");
            }
            if (i < shape[depth] - 1) {
                printf(", ");
            }
        }
    }
}

Tensor tensor_full(int dim,...)
{
	va_list arg;
	Tensor tensor;

	va_start(arg,dim);
	tensor.shape =  create_shape(arg,dim);
	tensor.num_dims = dim;
	tensor.strides = create_stride(tensor.num_dims, tensor.shape);
	add_options(arg,&tensor);
	tensor.data = (float *)create_empty_data(dim,tensor.shape);
	float fill_value = va_arg(arg,double);
	tensor_fill(&tensor,fill_value);
	va_end(arg);
	return tensor;
}

void print_tensor(Tensor tensor) 
{
    if (tensor.num_dims <= 0) {
        printf("Error: Tensor must have at least 1 dimension.\n");
        return;
    }

    printf("Tensor of shape (");
    for (int i = 0; i < tensor.num_dims; i++) {
        printf("%d", tensor.shape[i]);
        if (i < tensor.num_dims - 1) {
            printf(", ");
        }
    }
    printf("):\n");

    print_tensor_recursive(tensor.data, tensor.shape, tensor.num_dims, 0, 0);
    printf("\n");
}

void tensor_set_seed(unsigned int seed)
{
	srand(seed);
}
float	generate_random()
{
	return (float)rand() / (float)RAND_MAX;	
}

Tensor tensor_rand(int dim,...)
{
	va_list arg;
	Tensor tensor;

	va_start(arg,dim);
	tensor.shape =  create_shape(arg,dim);
	tensor.num_dims = dim;
	tensor.strides = create_stride(tensor.num_dims, tensor.shape);
	add_options(arg,&tensor);
	tensor.data = (float *)create_empty_data(dim,tensor.shape);
	float *data = tensor.data;
	for(int i = 0; i < tensor_entries_len(&tensor); i++)
	{
		data[i] = generate_random(); 
	}
	va_end(arg);
	return tensor;
}
