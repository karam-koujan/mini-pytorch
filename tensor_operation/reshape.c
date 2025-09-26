#include "headers/tensor.h"
#include "headers/print.h"


void    *copy_data(Tensor *a)
{
    
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
        result->data = copy_data(result);
    }
    return (result);
}