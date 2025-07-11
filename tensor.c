#include "tensor.h"

void    error_msg(char *msg)
{
    printf("\033[31mError: %s\033[0m\n", msg);
}

int sizeof_type(Dtype type)
{
    int size = -1;
    if (type == FLOAT32 || type == INT32)
        size = 4;
    else if (type == DOUBLE || type == INT64)
        size = 8;
    else
        return (error_msg("invalid type!"), size);
    return (size);
}

int64_t *create_shape(const int64_t *shape, int64_t ndim)
{
    int64_t *new_shape = (int64_t *)malloc(ndim  * sizeof(int64_t));
    if (!new_shape)
        return (error_msg("shape creation failed!"), NULL);
    memcpy(new_shape, shape, sizeof(int64_t) * ndim);
    return (new_shape);
}

int64_t *create_stride(const int64_t *shape, int64_t ndim, Dtype type)
{
    int size = sizeof_type(type);
    if (size == -1)
        return (NULL);
    int64_t *strides = (int64_t *)malloc(ndim  * sizeof(int64_t));
    if (!strides)
        return (error_msg("stride creation failed!"), NULL);
    strides[ndim - 1] = size;
    for (int i = ndim - 2 ; i >= 0; i--)
    {
        strides[i] = size * shape[i];
        size = strides[i];
    }
    return (strides);
}

int64_t calculate_size(const int64_t *shape, int64_t ndim)
{
    int size = 1;
    for (int i = 0; i < ndim; i++)
    {
        size*= shape[i];
    }
    return (size);
}

void    *create_zero_data(Dtype type, int size)
{
    int val_size = sizeof_type(type);
    if (val_size == -1)
        return (NULL);
    void *data = malloc(size * val_size);
    if (!data)
        return (error_msg("data creation failed!!"), NULL);
    for (int i = 0; i < size; i++)
    {
        if (type == FLOAT32)
            ((float *)data)[i] = 0.0F;
        else if (type == DOUBLE)
            ((double *)data)[i] = 0.0;
        else if (type == INT32)
            ((int *)data)[i] = 0;
        else if (type == INT64)
            ((int64_t *)data)[i] = 0L;
    }
    return (data);
}

Tensor *tensor_zeros(const int64_t *shape, int64_t ndim, Dtype type, Device device)
{
    if (shape == NULL)
        return (error_msg("invalid shape"), NULL);
    if (ndim <= 0)
        return (error_msg("invalid ndim"), NULL);
    Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));
    if (!tensor)
        return (error_msg("tensor creation failed!"), NULL);
    tensor->shape = create_shape(shape, ndim);
    tensor->strides = create_stride(shape, ndim, type);
    tensor->size = calculate_size(shape, ndim);
    tensor->data = create_zero_data(type, tensor->size);
    tensor->device = device;
    tensor->num_dims = ndim;
    if (!tensor->shape || !tensor->strides)
    {
        error_msg("tensor creation failed!");
        free(tensor);
        free(tensor->shape);
        free(tensor->strides);
    }
    return (tensor);
}

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
        print_tensor_nbr(data, index ,type);
    } else {
        for (int i = 0; i < shape[depth]; i++) {
            if (depth < num_dims - 1) {
                printf("[");
            }
            print_tensor_recursive(data, shape, strides, num_dims, index + i * strides[depth], depth + 1, type);

            if (depth < num_dims - 1) {
                printf("]\n");
            }

            if (i < shape[depth] - 1) {
                printf(",");
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
        printf("%lld", tensor->shape[i]);
        if (i < tensor->num_dims - 1) {
            printf(",");
        }
    }
    printf("):\n");

    print_tensor_recursive(tensor->data, tensor->shape, tensor->strides, tensor->num_dims, 0, 0, tensor->dtype);
    printf("\n");
}

int main()
{
    const int64_t shape[] = {3, 13 , 13};
    Tensor *t = tensor_zeros(shape, 3, DOUBLE, CPU);
    tensor_print(t);
}