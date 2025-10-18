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
    int i = a->num_dims - 1;
    int j = b->num_dims - 2 <= 0 ? 0 : b->num_dims - 2;
    if (a->shape[i] == b->shape[j])
        return 1;
    return (0);
}
void    float_matmul(Tensor *a_r, Tensor *b_r, Tensor *result);




/*
Tasks: 
    - we should handle multitype
    - we should handle broadcasting for matrix mutltiplication
    - we should handle tensor_reshaping for result
    - we have to free all the leaks.
*/


Tensor *tensor_matmul(Tensor*a, Tensor *b)
{
    if (!is_shape_allowed(a,b))
        return (error_msg("the shapes are not compatible for matmul operation"), NULL);
    Tensor *a_r = tensor_deep_copy(a);
    if (!a_r)
        return (NULL);
    Tensor *b_r = tensor_deep_copy(b);
     if (!b_r)
        return (tensor_free(a_r), NULL);

    int64_t a_cols = a->shape[a->num_dims - 1];
    int64_t a_rows = a->shape[a->num_dims - 2];
    int64_t b_cols = b->shape[b->num_dims - 1];
    int64_t b_rows = b->shape[b->num_dims - 2];
  
    const int64_t a_r_shape[] = {-1, a_rows, a_cols};
    a_r = tensor_reshape(a_r ,a_r_shape, 3);
    if (!a_r)
        return (tensor_free(a_r), tensor_free(b_r), NULL);
    const int64_t b_r_shape[] = {-1, b_rows, b_cols};
    b_r = tensor_reshape(b_r, b_r_shape, 3);
    if (!b_r)
        return (tensor_free(a_r), tensor_free(b_r), NULL);
    int64_t batch_size = a_r->shape[0];
    const int64_t result_shape[] = {batch_size, a_rows, b_cols};
    double d = 0;
    Tensor *result = tensor_full(result_shape, 3, a->dtype, a->device, &d);
    if (!result)
        return (tensor_free(a_r), tensor_free(b_r), NULL);
    if (a_r->shape[0] != b_r->shape[0])
        error_msg("something is not working well in reshape");
    float_matmul(a_r, b_r, result);

    return result;
    // then handle the first 3 specicifc cases
}

void float_matmul(Tensor *a_r, Tensor *b_r, Tensor *result)
{
    int64_t batch_size = a_r->shape[0];
    double f = 0;
    void *acc  = &f;
    for (int64_t b_idx = 0; b_idx < batch_size; b_idx++)
    {
        for (int64_t a_rows = 0; a_rows < a_r->shape[1]; a_rows++)
        {
            for (int64_t b_cols = 0; b_cols < b_r->shape[2]; b_cols++)
            {
                *(double *)acc = 0.0f;
                for (int64_t a_cols = 0; a_cols < a_r->shape[2]; a_cols++)
                {
                    int64_t a_idx = b_idx * (a_r->strides[0] / sizeof_type(a_r->dtype))
                                  + a_rows * (a_r->strides[1] / sizeof_type(a_r->dtype))
                                  + a_cols * (a_r->strides[2] / sizeof_type(a_r->dtype));

                    int64_t bt_idx = b_idx *  (b_r->strides[0] / sizeof_type(a_r->dtype))
                                   + a_cols * (b_r->strides[1] / sizeof_type(a_r->dtype))
                                   + b_cols * (b_r->strides[2] / sizeof_type(a_r->dtype));
                    switch(a_r->dtype)
                    {
                        case FLOAT32: *((float *)acc) += ((float *)a_r->data)[a_idx] * ((float *)b_r->data)[bt_idx];break;
                        case DOUBLE: *((double *)acc) += ((double *)a_r->data)[a_idx] * ((double *)b_r->data)[bt_idx];break;
                        case INT32: *((int *)acc) += ((int *)a_r->data)[a_idx] * ((int *)b_r->data)[bt_idx];break;
                        case INT64: *((int64_t *)acc) += ((int64_t *)a_r->data)[a_idx] * ((int64_t *)b_r->data)[bt_idx];break;
                    }
                }

                int64_t r_idx = b_idx *  (result->strides[0] / sizeof_type(result->dtype))
                              + a_rows * (result->strides[1] / sizeof_type(result->dtype))
                              + b_cols * (result->strides[2] / sizeof_type(result->dtype));
                switch(result->dtype)
                {
                    case FLOAT32: ((float *)result->data)[r_idx] = *(float *)acc;break;
                    case DOUBLE: ((double *)result->data)[r_idx] = *(double *)acc;break;
                    case INT32: ((int *)result->data)[r_idx] = *(int *)acc;break;
                    case INT64: ((int64_t *)result->data)[r_idx] = *(int64_t *)acc; break;
                }
            }
        }
    }
}

Tensor *tensor_mm(Tensor*a, Tensor *b);
