#include "../headers/tensor.h"
#include "../headers/print.h"



Tensor *tensor_copy(Tensor *a)
{
    if (a == NULL)
    {
        error_msg("you entred an empty tensor");
        return (NULL);
    }
    Tensor *r = malloc(sizeof(Tensor));
    if (!r)
        return (NULL);
    memcpy(r, a, sizeof(r));
    return (r);
}


Tensor *tensor_transpose(Tensor *a, int64_t dim0, int64_t dim1)
{
    if (a == NULL)
    {
        error_msg("you entred an empty tensor");
        return (NULL);
    }
    if (dim0 < 0 || dim1 < 0)
    {
        error_msg("you entered a negative dim");
        return (NULL);
    }
    if (dim0 > a->num_dims - 1 || dim1 > a->num_dims - 1)
    {
        error_msg("you entered a dim > tensor dims");
        return (NULL);       
    }
    Tensor *r = tensor_copy(a);
    if (!r)
        return (NULL);
    int64_t tmp = r->shape[dim0];
    r->shape[dim0] = r->shape[dim1];
    r->shape[dim1] = tmp;
    tmp = r->strides[dim0];
    r->strides[dim0] = r->strides[dim1];
    r->strides[dim1] = tmp;
    return (r);
}