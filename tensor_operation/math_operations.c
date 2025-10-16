#include "../headers/tensor.h"
#include "../headers/print.h"


Dtype promote_dtype(Dtype a, Dtype b)
{
    if (a > b)
        return a;
    return b;
}

void    pairwise_add(Tensor *a, Tensor *b, Tensor *r)
{
    for (int i = 0; i < a->size; i++)
    {
        if (r->dtype == FLOAT32)

            ((float *)r->data)[i] = ((float *)a->data)[i] + ((float *)b->data)[i];
        else if (r->dtype == DOUBLE)
            ((double *)r->data)[i] = ((double *)a->data)[i] + ((double *)b->data)[i];
        else if (r->dtype == INT32)
            ((int *)r->data)[i] = ((int *)a->data)[i] + ((int *)b->data)[i];
        else if (r->dtype == INT64)
            ((int64_t *)r->data)[i] = ((int64_t *)a->data)[i] + ((int64_t *)b->data)[i];
    }
}

void    pairwise_op(Tensor *a, Tensor *b, Tensor *r, char op)
{
    if (op == '+')
    {
        pairwise_add(a, b, r);
    }
}

Tensor *tensor_add(Tensor*a, Tensor *b)
{
    if (tensor_broadcast(a,b))
        return (NULL);

   // if they have not the same datatype promote datatype
   Dtype dtype = a->dtype;
    if (a->dtype != a->dtype)
    {
        dtype = promote_dtype(a->dtype, b->dtype);
    }
    Tensor *r = tensor_full(a->shape, a->num_dims, dtype, a->device, 0);
    if (!r)
        return (NULL);

    pairwise_op(a, b, r, '+');
    // create a tensor_full with the newshape

   // preform the operation, I think I should create a  shared helper function that do pairwise  operations
}


Tensor *tensor_sub(Tensor*a, Tensor *b);
Tensor *tensor_matmul(Tensor*a, Tensor *b);
Tensor *tensor_mul(Tensor*a, Tensor *b);
Tensor *tensor_mm(Tensor*a, Tensor *b);
