#include "../headers/tensor.h"
#include "../headers/print.h"

int is_tensor_broadcastable(Tensor *a, Tensor *b)
{
    int i = a->num_dims - 1;
    int j = b->num_dims - 1;
    int k = i > j ? j : i;
    while (k >= 0)
    {
        if (a->shape[i] != b->shape[j] && b->shape[j] != 1 && a->shape[i] != 1)
            return (error_msg("the tensors are not broadcastable") ,0);
        i--;
        j--;
        k--;
    }
    return (1);
}

int tensor_broadcast(Tensor *a, Tensor *b)
{
    if (!is_tensor_broadcastable(a, b))
        return (1);
    int ndim = b->num_dims > a->num_dims ? b->num_dims : a->num_dims;
    int i = a->num_dims - 1;
    int j = b->num_dims - 1;
    int k = i > j ? j : i;
    int64_t *shape_a = (int64_t *)malloc(ndim * sizeof(int64_t));
    int64_t *shape_b = (int64_t *)malloc(ndim * sizeof(int64_t));
    int64_t *stride_a = (int64_t *)malloc(ndim * sizeof(int64_t));
    int64_t *stride_b = (int64_t *)malloc(ndim * sizeof(int64_t));
    if (!shape_a || !shape_b || !stride_a || !stride_b)
        return (error_msg("error in creating shape in tensor_broadcast"), 1);
    memcpy(stride_a, a->strides, ndim * sizeof(int64_t));
    memcpy(stride_b, b->strides, ndim * sizeof(int64_t));
    memcpy(shape_a, a->shape, ndim * sizeof(int64_t));
    memcpy(shape_b, b->shape, ndim * sizeof(int64_t));
    while (k >= 0)
    {
        if (a->shape[i] != b->shape[j] && a->shape[i] == 1)
        {
            shape_a[j] = b->shape[j];
            stride_a[j] = 0;
            a->is_broadcasted = 1;
        }
        else if (a->shape[i] != b->shape[j] && b->shape[j] == 1)
        {
            shape_b[i] = a->shape[i];
            stride_b[i] = 0;
            b->is_broadcasted = 1;
        }
        else if (a->num_dims > b->num_dims)
        {
            shape_a[i] = a->shape[i];
            shape_b[i] = b->shape[j];
            stride_a[i] = a->strides[i];
            stride_b[i] = b->strides[j];
        }
        else
        {
                shape_a[j] = a->shape[i];
                shape_b[j] = b->shape[j];
                stride_a[j] = a->strides[i];
                stride_b[j] = b->strides[j];
        }
        k--;
        j--;
        i--; 
    }
    if (i >= 0)
    {
        b->is_broadcasted = 1;
        memcpy(shape_b, shape_a, (i + 1) * sizeof(int64_t));
    }
    else if (j >= 0)
    {
        a->is_broadcasted = 1;
        memcpy(shape_a, shape_b, (j + 1) * sizeof(int64_t));
    }
    for (int sa = 0; sa <= i; sa++)
        stride_b[sa] = 0;
    for (int sb = 0; sb <= j; sb++)
        stride_a[sb] = 0;
//     if (a->is_broadcasted)
//     {
//         a->prebroadcast_shape = a->shape;
//         a->prebroadcast_stride = a->strides;
//         a->prebroadcast_dims = a->num_dims;
//         a->shape = shape_a;
//         a->strides = stride_a;
//         a->num_dims = ndim;
//     }else
//     {
//         free(shape_a);
//         free(stride_a);
//     }
//     if (b->is_broadcasted)
//     {
//         b->prebroadcast_shape = b->shape;
//         b->prebroadcast_stride = b->strides;
//         b->prebroadcast_dims = b->num_dims;
//         b->shape = shape_b;
//         b->strides = stride_b;
//         b->num_dims = ndim;
//     }
// else
//     {
//         free(shape_b);
//         free(stride_b);
//     }
    a->shape = shape_a;
    a->strides = stride_a;
    a->num_dims = ndim;
    b->shape = shape_b;
    b->strides = stride_b;
    b->num_dims = ndim;
    return (0);
}
