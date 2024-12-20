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
	if (!shape)
		return NULL;
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
	if (!tensor)
		return ;
	tensor->device = CPU;
	tensor->dtype = FLOAT32;
	tensor->requires_grad = 1;

	char	*device = va_arg(arg,char *);
	char	*dtype = va_arg(arg,char *);
	int		requires_grad = va_arg(arg, int);
	if (requires_grad && requires_grad == 0)
	{
		tensor->requires_grad = 0;
	}
	if (device && !strcmp(device,"gpu"))
	{
		tensor->device = GPU;
	}
	if (dtype && !strcmp(dtype,"int"))
	{
		tensor->dtype = INT32;
	}
	va_end(arg);
}
float	*create_empty_data(int dim,int *shape)
{
	if (!shape)
		return NULL;
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
	if (!tensor)
		return -1;
	int count = 1;
	for(int i = 0; i < tensor->num_dims; i++)
	{
		count*=tensor->shape[i];
	}
	return count;
}

void	tensor_fill(Tensor *tensor, float num)
{
	if (!tensor)
		return ;
	int tensor_entries_num = tensor_entries_len(tensor);
	float *data = (float *)tensor->data;
	for(int i = 0; i < tensor_entries_num;i++)
	{
		data[i] = num;
	}
}

Tensor	 *tensor_empty(int dim,...)
{
	va_list arg;
	Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));

	va_start(arg,dim);
	tensor->shape =  create_shape(arg,dim);
	tensor->strides = create_stride(dim, tensor->shape);
	if (!tensor->shape || !tensor->strides)
	{
		free(tensor->shape);
		free(tensor->strides);
		return NULL;
	}
	tensor->num_dims = dim;
	int op = va_arg(arg, int);
	if (op > 0)
		add_options(arg,tensor);
	tensor->data = (float *)create_empty_data(dim,tensor->shape);
	tensor->is_leaf = 1;
	va_end(arg);
	return tensor;
}

Tensor *tensor_zeros(int dim,...)
{
	va_list arg;
	Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));

	va_start(arg,dim);
	tensor->shape =  create_shape(arg,dim);
	tensor->strides = create_stride(dim, tensor->shape);
	if (!tensor->shape || !tensor->strides)
	{
		free(tensor->shape);
		free(tensor->strides);
		return NULL;
	}
	tensor->num_dims = dim;
	int op = va_arg(arg, int);
	if (op > 0)
		add_options(arg,tensor);
	tensor->data = (float *)create_empty_data(dim,tensor->shape);
	tensor->is_leaf = 1;
	tensor_fill(tensor,0);
	va_end(arg);
	return tensor;
}



Tensor *tensor_ones(int dim,...)
{
	va_list arg;
	Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));

	va_start(arg,dim);
	tensor->shape =  create_shape(arg,dim);
	tensor->strides = create_stride(dim, tensor->shape);
	if (!tensor->shape || !tensor->strides)
	{
		free(tensor->shape);
		free(tensor->strides);
		return NULL;
	}
	tensor->num_dims = dim;
	int op = va_arg(arg, int);
	if (op > 0)
		add_options(arg,tensor);
	tensor->data = (float *)create_empty_data(dim,tensor->shape);
	tensor->is_leaf = 1;
	tensor_fill(tensor,1);
	va_end(arg);
	return tensor;
}
void print_tensor_recursive(float *data, int *shape, int *strides, int num_dims, int index, int depth) {
    if (depth == num_dims) {
        printf("%f", data[index]);
    } else {
        for (int i = 0; i < shape[depth]; i++) {
            if (depth < num_dims - 1) {
                printf("[");
            }
            print_tensor_recursive(data, shape, strides, num_dims, index + i * strides[depth], depth + 1);

            if (depth < num_dims - 1) {
                printf("]\n");
            }

            if (i < shape[depth] - 1) {
                printf(", ");
            }
        }
    }
}



void tensor_print(Tensor *tensor) 
{
	if (!tensor)
		return ;
    if (tensor->num_dims <= 0) {
        printf("Error: Tensor must have at least 1 dimension.\n");
        return;
    }

    printf("Tensor of shape (");
    for (int i = 0; i < tensor->num_dims; i++) {
        printf("%d", tensor->shape[i]);
        if (i < tensor->num_dims - 1) {
            printf(", ");
        }
    }
    printf("):\n");

    print_tensor_recursive(tensor->data, tensor->shape, tensor->strides, tensor->num_dims, 0, 0);
    printf("\n");
}

Tensor *tensor_full(int dim,...)
{
	va_list arg;
	Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));

	va_start(arg,dim);
	tensor->shape =  create_shape(arg,dim);
	tensor->strides = create_stride(dim, tensor->shape);
	if (!tensor->shape || !tensor->strides)
	{
		free(tensor->shape);
		free(tensor->strides);
		va_end(arg);
		return NULL;
	}
	tensor->num_dims = dim;
	float fill_value = va_arg(arg,double);
	int op = va_arg(arg, int);
	if (op > 0)
		add_options(arg,tensor);
	
	tensor->data = (float *)create_empty_data(dim,tensor->shape);
	tensor->is_leaf = 1;
	tensor_fill(tensor,fill_value);
	va_end(arg);
	return tensor;
}


void tensor_set_seed(unsigned int seed)
{
	srand(seed);
}
float	generate_random()
{
	return (float)rand() / (float)RAND_MAX;	
}

Tensor *tensor_rand(int dim,...)
{
	va_list arg;
	Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));

	va_start(arg,dim);
	tensor->shape =  create_shape(arg,dim);
	tensor->strides = create_stride(dim, tensor->shape);
	if (!tensor->shape || !tensor->strides)
	{
		free(tensor->shape);
		free(tensor->strides);
		va_end(arg);
		return NULL;
	}
	tensor->num_dims = dim;
	int op = va_arg(arg, int);
	if (op > 0)
		add_options(arg,tensor);
	float *data = (float *)create_empty_data(dim,tensor->shape);
	for(int i = 0; i < tensor_entries_len(tensor); i++)
	{
		data[i] = generate_random(); 
	}
	tensor->data = data;
	tensor->is_leaf = 1;
	va_end(arg);
	return tensor;
}

ssize_t	tensor_size(Tensor *a, ssize_t num_dims)
{
	ssize_t size = 1;
	if (num_dims > a->num_dims)
	{
		fprintf(stderr,"num_dims is bigger than the tensor dims!!");
		return -1;
	}
	for(int i = 0; i < num_dims; i++)
	{
		size*= a->shape[i];
	}
	return size;
}
float	*tensor_contigous_data(Tensor *a, int *new_shape)
{
	int size = tensor_size(a,a->num_dims);
	float *data = malloc(size * sizeof(float));
	float *old_data = a->data;
	if (!data)
		return NULL;
	for(int i = 0; i < size; i++)
	{
		int tmp = i;
		int offset = 0;
		for(int j = a->num_dims - 1; j >= 0; j--)
		{
			int coord = tmp % new_shape[j];
			offset+= coord * a->strides[j];
			tmp /= new_shape[j];
		}
		data[i] = old_data[offset];
	}
	return data;
}

float	*tensor_contigous(Tensor *a, int *new_shape)
{
	int is_contigious = 1;
	int prev_stride = a->strides[0];
	for(int i = 1; i < a->num_dims;i++)
	{
		if (a->strides[i] == 0)
		{
			is_contigious = 0;
			break;
		}
		if (a->strides[i] > prev_stride)
		{
			is_contigious = 0;
			break;
		}
		prev_stride = a->strides[i];
	}
	if (is_contigious == 1)
		return a->data;
	return tensor_contigous_data(a, new_shape);
}

