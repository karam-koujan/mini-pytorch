#include "../headers/tensor.h"
#include "../headers/print.h"


Dtype promote_dtype(Dtype a, Dtype b)
{
    if (a == DOUBLE || b == DOUBLE)
        return DOUBLE;
    if (a == FLOAT32 || b == FLOAT32)
        return FLOAT32;
    if (a == INT64 || b == INT64)
        return INT64;
    return INT32;
}

void   *promote_data(Tensor *a, Dtype dtype)
{
    void *data;
    switch(dtype)
    {
        case DOUBLE:
            data = (double *)malloc(a->size * sizeof(double));
            break;
        case FLOAT32:
            data = (float *)malloc(a->size * sizeof(float));
            break;
        case INT64:
            data = (int64_t *)malloc(a->size * sizeof(int64_t));
            break;
        case INT32:
            data = (int *)malloc(a->size * sizeof(int));
            break;         
    }
    if (!data)
        return (NULL);
    for(int i = 0; i < a->size; i++)
    {
        double val = 0.0;
        switch (a->dtype)
        {
            case DOUBLE:  val = ((double *)a->data)[i]; break;
            case FLOAT32: val = ((float *)a->data)[i]; break;
            case INT64:   val = (double)((int64_t *)a->data)[i]; break;
            case INT32:   val = (double)((int *)a->data)[i]; break;
        }

        switch (dtype)
        {
            case DOUBLE:  ((double *)data)[i] = val; break;
            case FLOAT32: ((float *)data)[i] = (float)val; break;
            case INT64:   ((int64_t *)data)[i] = (int64_t)val; break;
            case INT32:   ((int *)data)[i] = (int)val; break;
        }
    }
    return (data);
}



void    pairwise_add(void *a, void *b, Tensor *r)
{
    for (int i = 0; i < r->size; i++)
    {
        if (r->dtype == FLOAT32)

            ((float *)r->data)[i] = ((float *)a)[i] + ((float *)b)[i];
        else if (r->dtype == DOUBLE)
            ((double *)r->data)[i] = ((double *)a)[i] + ((double *)b)[i];
        else if (r->dtype == INT32)
            ((int *)r->data)[i] = ((int *)a)[i] + ((int *)b)[i];
        else if (r->dtype == INT64)
            ((int64_t *)r->data)[i] = ((int64_t *)a)[i] + ((int64_t *)b)[i];
    }
}
void    pairwise_sub(void *a, void *b, Tensor *r)
{
    for (int i = 0; i < r->size; i++)
    {
        if (r->dtype == FLOAT32)

            ((float *)r->data)[i] = ((float *)a)[i] - ((float *)b)[i];
        else if (r->dtype == DOUBLE)
            ((double *)r->data)[i] = ((double *)a)[i] - ((double *)b)[i];
        else if (r->dtype == INT32)
            ((int *)r->data)[i] = ((int *)a)[i] - ((int *)b)[i];
        else if (r->dtype == INT64)
            ((int64_t *)r->data)[i] = ((int64_t *)a)[i] - ((int64_t *)b)[i];
    }
}
void    pairwise_div(void *a, void *b, Tensor *r)
{
    for (int i = 0; i < r->size; i++)
    {
        if (r->dtype == FLOAT32)

            ((float *)r->data)[i] = ((float *)a)[i] / ((float *)b)[i];
        else if (r->dtype == DOUBLE)
            ((double *)r->data)[i] = ((double *)a)[i] / ((double *)b)[i];
        else if (r->dtype == INT32)
            ((int *)r->data)[i] = ((int *)a)[i] / ((int *)b)[i];
        else if (r->dtype == INT64)
            ((int64_t *)r->data)[i] = ((int64_t *)a)[i] / ((int64_t *)b)[i];
    }
}

void    pairwise_mul(void *a, void *b, Tensor *r)
{
    for (int i = 0; i < r->size; i++)
    {
        if (r->dtype == FLOAT32)

            ((float *)r->data)[i] = ((float *)a)[i] * ((float *)b)[i];
        else if (r->dtype == DOUBLE)
            ((double *)r->data)[i] = ((double *)a)[i] * ((double *)b)[i];
        else if (r->dtype == INT32)
            ((int *)r->data)[i] = ((int *)a)[i] * ((int *)b)[i];
        else if (r->dtype == INT64)
            ((int64_t *)r->data)[i] = ((int64_t *)a)[i] * ((int64_t *)b)[i];
    }
}

void    pairwise_op(Tensor *a, Tensor *b, Tensor *r, char op)
{
    if (op == '+')
        pairwise_add(a, b, r);
    else if (op == '-')
        pairwise_sub(a, b, r);
    else if (op == '/')
        pairwise_div(a, b, r);
    else if (op == '*')
        pairwise_mul(a,b, r);
}

Tensor *tensor_add(Tensor*a, Tensor *b)
{
    if (tensor_broadcast(a,b))
        return (NULL);
    Dtype dtype = a->dtype;
    void *data_a;
    void *data_b;

    dtype = promote_dtype(a->dtype, b->dtype);
    data_a = promote_data(a, dtype);
    if (!data_a)
        return (NULL);
    data_b = promote_data(b, dtype);
    if (!data_b)
        return (free(data_a), NULL);
    int val = 0;
    void *val_ptr = &val;
    Tensor *r = tensor_full(a->shape, a->num_dims, dtype, a->device, val_ptr);
    if (!r)
        return (NULL);
    pairwise_op(data_a, data_b, r, '+');
    free(data_a);
    free(data_b);
    tensor_unbroadcast(a);
    tensor_unbroadcast(b);
    return (r);
}


Tensor *tensor_sub(Tensor*a, Tensor *b)
{
    if (tensor_broadcast(a,b))
        return (NULL);
    Dtype dtype = a->dtype;
    void *data_a;
    void *data_b;
    dtype = promote_dtype(a->dtype, b->dtype);
    data_a = promote_data(a, dtype);
    if (!data_a)
        return (NULL);
    data_b = promote_data(b, dtype);
    if (!data_b)
        return (free(data_a), NULL);
    int val = 0;
    void *val_ptr = &val;
    Tensor *r = tensor_full(a->shape, a->num_dims, dtype, a->device, val_ptr);
    if (!r)
        return (NULL);
    pairwise_op(data_a, data_b, r, '-');
    free(data_a);
    free(data_b);
    tensor_unbroadcast(a);
    tensor_unbroadcast(b);
    return (r);
}

Tensor *tensor_div(Tensor*a, Tensor *b)
{
    if (tensor_broadcast(a,b))
        return (NULL);
    Dtype dtype = a->dtype;
    void *data_a;
    void *data_b;

    dtype = promote_dtype(a->dtype, b->dtype) == FLOAT32 ? FLOAT32 : DOUBLE;
    data_a = promote_data(a, dtype);
    if (!data_a)
        return (NULL);
    data_b = promote_data(b, dtype);
    if (!data_b)
        return (free(data_a), NULL);
    int val = 0;
    void *val_ptr = &val;
    Tensor *r = tensor_full(a->shape, a->num_dims, dtype, a->device, val_ptr);
    if (!r)
        return (NULL);
    pairwise_op(data_a, data_b, r, '/');
    free(data_a);
    free(data_b);
    tensor_unbroadcast(a);
    tensor_unbroadcast(b);
    return (r);
}

Tensor *tensor_mul(Tensor*a, Tensor *b)
{
    if (tensor_broadcast(a,b))
        return (NULL);
    Dtype dtype = a->dtype;
    void *data_a;
    void *data_b;

    dtype = promote_dtype(a->dtype, b->dtype);
    data_a = promote_data(a, dtype);
    if (!data_a)
        return (NULL);
    data_b = promote_data(b, dtype);
    if (!data_b)
        return (free(data_a), NULL);
    
    int val = 0;
    void *val_ptr = &val;
    Tensor *r = tensor_full(a->shape, a->num_dims, dtype, a->device, val_ptr);
    if (!r)
        return (NULL);
    pairwise_op(data_a, data_b, r, '*');
    free(data_a);
    free(data_b);
    tensor_unbroadcast(a);
    tensor_unbroadcast(b);
    return (r);
}

int is_shape_allowed(Tensor*a, Tensor *b)
{
    int i = a->size - 1;
    int j = b->size - 2 <= 0 ? 0 : b->size - 2;
    if (a->shape[i] == b->shape[j])
        return 1;
    return (0);
}

Tensor *tensor_matmul(Tensor*a, Tensor *b)
{
    // check if the matrix dim is correct
    if (is_shape_allowed(a,b))
        return (error_msg("the shapes are not compatible for matmul operation"), NULL);
    // check if the tensors are broadcastable

    // reshape the tensors so it have (batch, n, m)

    // do the calculation 

    // then handle the first 3 specicifc cases
}


Tensor *tensor_mm(Tensor*a, Tensor *b);
