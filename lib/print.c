#include "../headers/print.h"

void print_tensor_nbr(void *data, int index, Dtype type)
{
    if (type == FLOAT32)
       printf("%.2f", ((float *)data)[index]);
    else if (type == DOUBLE)
        printf("%.2f", ((double *)data)[index]);
    else if (type == INT32)
        printf("%i", ((int *)data)[index]);
    else if (type == INT64)
        printf("%lli", ((long long *)data)[index]);
}

// Corrected recursive printing function
void print_tensor_recursive(void *data, int64_t *shape, int64_t *strides, int num_dims, int index, int depth, Dtype type) {
    if (depth == num_dims) {
        // Base case: we have recursed through all dimensions, print the number.
        print_tensor_nbr(data, index, type);
        return;
    }

    printf("[");
    for (int i = 0; i < shape[depth]; i++) {
        int element_stride = strides[depth] / sizeof_type(type);
        print_tensor_recursive(data, shape, strides, num_dims, index + i * element_stride, depth + 1, type);
        
        if (i < shape[depth] - 1) {
            printf(",");
            // Add a newline and indentation for better readability on outer dimensions
            if (depth < num_dims - 2) {
                printf("\n");
                for (int d = 0; d <= depth; d++) {
                    printf(" ");
                }
            } else {
                printf(" "); // Just a space for the innermost dimension
            }
        }
    }
    printf("]");
}

void print_shape(int64_t *shape, int dims)
{
    printf("Tensor of shape (");
    for (int i = 0; i < dims; i++) {
        printf("%lld", shape[i]);
        if (i < dims - 1) {
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
    else if (type == GPU) // Corrected from DOUBLE
        printf("device: gpu\n");
}

void print_strides(int64_t *strides, int dims)
{
    printf("Tensor of strides (");
    for (int i = 0; i < dims; i++) {
        printf("%lld", strides[i]);
        if (i < dims - 1) {
            printf(",");
        }
    }
    printf("):\n");
}

void tensor_print(Tensor *tensor) 
{
	if (!tensor) {
		printf("NULL Tensor\n");
		return;
    }
    if (tensor->num_dims <= 0 && tensor->size == 1) { // Handle scalar
        print_shape(tensor->shape, tensor->num_dims);
        print_tensor_nbr(tensor->data, 0, tensor->dtype);
        printf("\n");
        return;
    }
    if (tensor->num_dims <= 0) {
        printf("Error: Tensor has no dimensions and is not a scalar.\n");
        return;
    }
    print_shape(tensor->shape, tensor->num_dims);
    print_tensor_recursive(tensor->data, tensor->shape, tensor->strides, tensor->num_dims, 0, 0, tensor->dtype);
    printf("\n");
}

void    error_msg(char *msg)
{
    printf("\033[31mError: %s\033[0m\n", msg);
}
