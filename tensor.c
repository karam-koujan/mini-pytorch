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

    return (NULL);
}


int main()
{
    tensor_zeros(NULL, -5, 0 , 0);
}