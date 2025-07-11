#include "../headers/print.h"

void print_tensor_nbr(void *data, int index, Dtype type)
{
    if (type == FLOAT32)
       printf("%2.f", ((float *)data)[index]);
    else if (type == DOUBLE)
        printf("%2.f", ((double *)data)[index]);
    else if (type == INT32)
        printf("%i", ((int *)data)[index]);
    else if (type == INT64)
        printf("%lli", ((int64_t *)data)[index]);
}

void print_tensor_recursive(void *data, int64_t *shape, int64_t *strides, int num_dims, int index, int depth, Dtype type) {
    if (depth == num_dims) {
        print_tensor_nbr(data, index, type);
    } else {
        for (int i = 0; i < shape[depth]; i++) {
            if (depth < num_dims - 1) {
                printf("[");
            }
            int element_stride = strides[depth] / sizeof_type(type);
            print_tensor_recursive(data, shape, strides, num_dims, index + i * element_stride, depth + 1, type);

            if (depth < num_dims - 1) {
                printf("]\n");
            }

            if (i < shape[depth] - 1) {
                printf(",");
            }
        }
    }
}

void print_shape(Tensor *tensor)
{
    printf("Tensor of shape (");
    for (int i = 0; i < tensor->num_dims; i++) {
        printf("%lld", tensor->shape[i]);
        if (i < tensor->num_dims - 1) {
            printf(",");
        }
    }
    printf("):\n");
}

void print_type(Dtype type)
{
    if (type == FLOAT32)
        printf("dtype: float32\n");
    else if (type == DOUBLE)
        printf("dtype: float64\n");
    else if (type == INT32)
        printf("dtype: int32\n");
    else if (type == INT64)
        printf("dtype: int64\n");
}

void print_device(Device type)
{
    if (type == CPU)
        printf("device: cpu\n");
    else if (type == DOUBLE)
        printf("device: gpu\n");
}

void print_strides(Tensor *tensor)
{
    printf("Tensor of strides (");
    for (int i = 0; i < tensor->num_dims; i++) {
        printf("%lld", tensor->strides[i]);
        if (i < tensor->num_dims - 1) {
            printf(",");
        }
    }
    printf("):\n");
}

void tensor_print(Tensor *tensor) 
{
	if (!tensor)
		return ;
    if (tensor->num_dims <= 0) {
        printf("Error: Tensor must have at least 1 dimension.\n");
        return;
    }
    print_shape(tensor);
    print_tensor_recursive(tensor->data, tensor->shape, tensor->strides, tensor->num_dims, 0, 0, tensor->dtype);
    printf("\n");
}

void    error_msg(char *msg)
{
    printf("\033[31mError: %s\033[0m\n", msg);
}