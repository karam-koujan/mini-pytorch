#include "../headers/tensor.h"


void    tensor_free(Tensor *a)
{
    if (!a)
        return ;
    free(a->data);
    a->data = NULL;
    free(a->shape);
    a->shape = NULL;
    tensor_free(a->grad);
    a->grad = NULL;
    free(a->strides);
    a->strides = NULL;
    if (a->is_broadcasted)
    {
        free(a->prebroadcast_shape);
        a->prebroadcast_shape = NULL;
        free(a->prebroadcast_stride);
        a->prebroadcast_stride = NULL;
    }
    if (a->grad_fn)
    {
        Grad_Node *n = a->grad_fn;
        tensor_free(n->grad);
        n->grad = NULL;
        tensor_free(n->broadcasted_tensor_a);
        n->broadcasted_tensor_a = NULL;
        tensor_free(n->broadcasted_tensor_b);
        n->broadcasted_tensor_b = NULL;
        free(n);
    }
    a->grad_fn = NULL;
    free(a);
}

void    tensor_free_after_reshape(Tensor *a)
{
    if (!a)
        return ;
    if (a->data && !is_contigious(a))
    {
        free(a->data);
        a->data = NULL;
    }
    free(a->shape);
    a->shape = NULL;
    free(a->strides);
    a->strides = NULL;

    free(a);
}