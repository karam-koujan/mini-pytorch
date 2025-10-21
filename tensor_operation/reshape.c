#include "../headers/tensor.h"
#include "../headers/print.h"

void    fill_data(void *data, int offset, Dtype dtype, void *value)
{
    if (dtype == FLOAT32)
        ((float *)data)[offset] = *((float *)value);
    else if (dtype == DOUBLE)
        ((double *)data)[offset] = *((double *)value);
    else if (dtype == INT32)
        ((int *)data)[offset] = *((int *)value);
    else if (dtype == INT64)
        ((int64_t *)data)[offset] = *((int64_t *)value); 
}

void    *copy_contigious_data(Tensor *a, int size)
{
    int val_size = sizeof_type(a->dtype);
    if (val_size == -1)
        return (NULL);
    void *data = malloc(size * val_size);
    if (!data)
        return (error_msg("data creation failed!!"), NULL);
    int coord = 0;
    int tmp = 0;
    int offset;
    for (int i = 0; i < size; i++)
    {
        tmp = i;
        offset = 0;
        for (int j = a->num_dims - 1; j >= 0; j--)
        {
            coord = tmp % a->shape[j];
            tmp /= a->shape[j];
            offset += coord * a->strides[j];    
        }

        fill_data(data, i, a->dtype, (char *)a->data + offset);
    }
    return (data);
}


Tensor  *tensor_reshape(Tensor *a, const int64_t *view, int64_t new_ndim)
{
    if (!is_view_allowed(view, new_ndim))
        return (NULL);
    int64_t *new_shape = infer_shape_from_view(a, view, new_ndim);
    if (!new_shape)
        return (NULL);
    int64_t *new_stride = create_stride(new_shape, new_ndim, a->dtype);
    if (!new_stride)
        return (NULL);
    Tensor *result = malloc(sizeof(Tensor));
    if (!result)
        return (NULL);
    memcpy(result, a, sizeof(Tensor));
    result->num_dims = new_ndim;
    result->shape = new_shape;
    result->strides = new_stride;
    if (!is_contigious(a))
    {
            result->data = copy_contigious_data(result, result->size);
    }
    return (result);
}