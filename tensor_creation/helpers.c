#include "../headers/tensor.h"
#include "../headers/print.h"

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

int64_t calculate_size(const int64_t *shape, int64_t ndim)
{
    int size = 1;
    for (int i = 0; i < ndim; i++)
    {
        size*= shape[i];
    }
    return (size);
}

void tensor_set_seed(unsigned int seed)
{
	srand(seed);
}

float	generate_random()
{
     float r = (float)rand() / (float)RAND_MAX;
    
	return r;	
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
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return (strides);
}

Tensor *tensor_deep_copy(Tensor *a)
{
    if (a == NULL)
    {
        error_msg("you entred an empty tensor");
        return (NULL);
    }
    Tensor *r = malloc(sizeof(Tensor));
    if (!r)
        return (NULL);
    memcpy(r, a, sizeof(Tensor));
    void *data = malloc(a->size * sizeof(a->dtype));
    if (!data)
    {
        r->data = NULL;
        return (tensor_free(r), NULL);
    }
    memcpy(data, a->data, a->size * sizeof(a->dtype));
    r->data = data;
    return (r);
}

