#include "tensor.h"

void    error_msg(char *msg)
{
    printf("\033[31mError: %s\033[0m\n", msg);
}

int64_t *create_shape(const int64_t *shape, int64_t ndim)
{
    int64_t *new_shape = (int64_t *)malloc(ndim  * sizeof(int64_t));
    if (!new_shape)
        return (error_msg("shape creation failed!"), NULL);
    memcpy(new_shape, shape, sizeof(int64_t));
    return (new_shape);
}

int64_t *create_stride(const int64_t *shape, int64_t ndim, Dtype type)
{
    int size;
    int64_t *strides = (int64_t *)malloc(ndim  * sizeof(int64_t));
    if (!strides)
        return (error_msg("stride creation failed!"), NULL);
    if (type == FLOAT32 || type == INT32)
        size = 4;
    else if (type == DOUBLE || type == INT64)
        size = 8;
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
    int val_size;
    if (type == FLOAT32 || type == INT32)
        val_size = 4;
    else if (type == DOUBLE || type == INT64)
        val_size = 8;
    void *data = malloc(size * val_size);
    if (!data)
        return (error_msg("data creation failed!!"), NULL);
    
    for (int i = 0; i < size; i++)
    {
        if (type == FLOAT32)
            data[i] = 0.0F; 
        else if (type == DOUBLE)
            data[i] = 0.0;
        else if (type == INT32)
            data[i] = 0;
        else if (type == INT64)
            data[i] = 0L;      
    }

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
    tensor->data = create_data(type, tensor->size, 0);
    if (!tensor->shape || !tensor->strides)
    {
        error_msg("tensor creation failed!");
        free(tensor);
        free(tensor->shape);
        free(tensor->strides);
    }
    return (NULL);
}


int main()
{
    tensor_zeros(NULL, -5, 0 , 0);
}