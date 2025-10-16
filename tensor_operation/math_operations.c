#include "../headers/tensor.h"
#include "../headers/print.h"



void   change_dtype(Tensor *a, Dtype dtype)
{
    void *data = a->data;
    for (int i = 0; i < a->size; i++)
    {
        if (dtype == FLOAT32)
            ((float *)data)[i] = ((float *)data)[i];
        else if (dtype == DOUBLE)
            ((double *)data)[i] = ((double *)data)[i];
        else if (dtype == INT32)
            ((int *)data)[i] = ((int *)data)[i];
        else if (dtype == INT64)
            ((int64_t *)data)[i] = ((int64_t *)data)[i];
    }
}


Dtype promote_dtype(Tensor *a, Tensor *b)
{
    if (a->dtype > b->dtype)
    {
        change_dtype(b, a->dtype);
        return a->dtype;
    }
    change_dtype(a, b->dtype);
    return b->dtype;
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
    if (a->dtype != b->dtype)
    {
        dtype = promote_dtype(a, b);
    }
    int val = 0;
    void *val_ptr = &val;
    Tensor *r = tensor_full(a->shape, a->num_dims, dtype, a->device, val_ptr);
    if (!r)
        return (NULL);

    pairwise_op(a, b, r, '+');
    return (r);
}


Tensor *tensor_sub(Tensor*a, Tensor *b);
Tensor *tensor_matmul(Tensor*a, Tensor *b);
Tensor *tensor_mul(Tensor*a, Tensor *b);
Tensor *tensor_mm(Tensor*a, Tensor *b);
