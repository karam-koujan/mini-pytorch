#include "../headers/tensor.h"


void    tensor_free(Tensor *a)
{
    if (!a)
        return ;
    free(a->data);
    a->data = NULL;
    free(a->shape);
    free(a->grad);
    free(a->strides);
    if (a->is_broadcasted)
    {
        free(a->prebroadcast_shape);
        free(a->prebroadcast_stride);
    }
    free(a);
}