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

Tensor	 *tensor_empty(int dim,int *shape,...)
{
	va_list arg;
	Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));

	va_start(arg,shape);
	tensor->shape = (int *)malloc(dim * sizeof(int));
	if (!tensor->shape) return NULL;
	memcpy(tensor->shape, shape, dim * sizeof(int));	
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
	tensor->size = tensor_entries_len(tensor);
	tensor->requires_grad = 0;
	va_end(arg);
	return tensor;
}

Tensor *tensor_zeros(int dim,int *shape,...)
{
	va_list arg;
	Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));

	va_start(arg,shape);
	tensor->shape = (int *)malloc(dim * sizeof(int));
	if (!tensor->shape) return NULL;
	memcpy(tensor->shape, shape, dim * sizeof(int));		
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
	tensor->requires_grad = 0;
	tensor->size = tensor_entries_len(tensor);
	tensor_fill(tensor,0);
	va_end(arg);
	return tensor;
}



Tensor *tensor_ones(int dim,int *shape,...)
{
	va_list arg;
	Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));

	va_start(arg,shape);
	tensor->shape = (int *)malloc(dim * sizeof(int));
	if (!tensor->shape) return NULL;
	memcpy(tensor->shape, shape, dim * sizeof(int));	
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
	tensor->requires_grad = 0;
	tensor->size = tensor_entries_len(tensor);
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

Tensor *tensor_full(int dim,int *shape,...)
{
	va_list arg;
	Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));

	va_start(arg,shape);
	tensor->shape = (int *)malloc(dim * sizeof(int));
	if (!tensor->shape) return NULL;
	memcpy(tensor->shape, shape, dim * sizeof(int));
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
	tensor->size = tensor_entries_len(tensor);
	tensor->requires_grad = 0;

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

Tensor *tensor_rand(int dim,int *shape,...)
{
	va_list arg;
	Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));

	va_start(arg,shape);
	tensor->shape = (int *)malloc(dim * sizeof(int));
	if (!tensor->shape) return NULL;
	memcpy(tensor->shape, shape, dim * sizeof(int));	
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
	tensor->size = tensor_entries_len(tensor);
	tensor->requires_grad = 0;
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
		data[i] = old_data[offset % a->size];
	}
	return data;
}

int tensor_is_contigious(Tensor *a)
{
	int is_contigious = 1;
	int prev_stride = a->strides[0];
	for(int i = 0; i < a->num_dims;i++)
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
	return is_contigious;
}

float	*tensor_contigous(Tensor *a, int *new_shape)
{
	return tensor_contigous_data(a, new_shape);
}


void	tensor_set_require_grad(Tensor *a, int require_grad)
{
	if (require_grad == 1)
	{
		int *shape = malloc(a->num_dims * sizeof(int));
		memcpy(shape, a->shape, a->num_dims * sizeof(int));
		a->grad = tensor_zeros(a->num_dims,shape,0);
		a->requires_grad = 1;
	}
	if (require_grad == 0)
	{
		free(a->grad);
		a->requires_grad = 0;
	}
}

Tensor *tensor_detach(Tensor *a)
{
	Tensor *copy = malloc(sizeof(Tensor));
    if (!copy) return NULL; 
    *copy = *a;
    copy->requires_grad = 0;
    copy->grad_fn = NULL;

    return copy;
}

Tensor *tensor_clone(Tensor *a)
{
    Tensor *copy = malloc(sizeof(Tensor));
    if (!copy) return NULL;
    *copy = *a;

    copy->data = malloc(a->size * sizeof(float)); 
    if (!copy->data) {
        free(copy);
        return NULL;
    }
    memcpy(copy->data, a->data, a->size * sizeof(float));

    return copy;
}
int calculate1DIndex(int *indices, int *dims, int n) {
    int index = 0, multiplier = 1;
    for (int i = n - 1; i >= 0; i--) {
        index += indices[i] * multiplier;
        multiplier *= dims[i];
    }
    return index;
}

// Recursive function to flatten the N-dimensional array

float	*flattenNDArray(float *input, int *dims, int n, int size) {
    int totalElements = 1;
	float *output = malloc(size * sizeof(float));
	if (!output)
	{
		return NULL;
	}
    // Calculate total number of elements
    for (int i = 0; i < n; i++) {
        totalElements *= dims[i];
    }

    // Copy elements to the output array
    for (int i = 0; i < totalElements; i++) {
        output[i] = input[i];
    }
	return output;
}
Tensor *tensor_tensor(void *data, int *shape, int dims)
{
	Tensor *res = tensor_ones(dims,shape,0);

    float *output = flattenNDArray((float *)data, res->shape, dims,res->size);
	res->data = output;
	return res;
}
