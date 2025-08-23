#include "../headers/tensor.h"


void    tensor_free(Tensor *a)
{
    if (!a)
        return ;
    free(a->data);
    free(a->shape);
    free(a->grad);
    free(a->strides);
    free(a);
}