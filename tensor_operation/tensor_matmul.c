#include "../headers/tensor.h"
#include "../headers/print.h"


void tensor_free_view(Tensor *view) {
    if (!view) return;

    free(view->shape);
    free(view->strides);
    
    if (view->is_broadcasted) {
        free(view->prebroadcast_shape);
        free(view->prebroadcast_stride);
    }
    
    free(view);
}

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
    if (a_r->num_dims == 2 && b_r->num_dims == 2)
    {
      return (tensor_free(a_r), tensor_free(b_r), tensor_mm(a, b));
    }
    Grad_Node *grad_fn = a->requires_grad || b->requires_grad ? create_matmul_node(a,b) : NULL;
    if (a->num_dims >= 3 || b->num_dims >= 3) {
        if (tensor_matmul_broadcast(a_r, b_r)) {
            tensor_free(a_r);
            tensor_free(b_r);
            return NULL;
        }
        tensor_contigous_broadcast(a_r);
        tensor_contigous_broadcast(b_r);
        if (grad_fn)
        {
            grad_fn->broadcasted_tensor_a = tensor_deep_copy(a_r);
            if (!grad_fn->broadcasted_tensor_a)
                return (tensor_free(a_r), tensor_free(b_r), NULL);
            grad_fn->broadcasted_tensor_b = tensor_deep_copy(b_r);
            if (!grad_fn->broadcasted_tensor_b)
                return (tensor_free(a_r), tensor_free(b_r), NULL);
        }
    }
    int64_t a_cols = a->shape[a->num_dims - 1];
    int64_t a_rows = a->shape[a->num_dims - 2];
    int64_t b_cols = b->shape[b->num_dims - 1];
    int64_t b_rows = b->shape[b->num_dims - 2];
    int64_t *final_shape = calloc(a_r->num_dims, sizeof(int64_t));
    int final_dim = a_r->num_dims;
    if (!final_shape)
        return (tensor_free(a_r), tensor_free(b_r), NULL);
    memcpy(final_shape, a_r->shape, sizeof(int64_t) * a_r->num_dims);
    const int64_t a_r_shape[] = {-1, a_rows, a_cols};
    Tensor *a_f = tensor_reshape(a_r ,a_r_shape, 3);
    if (!a_f)
        return (tensor_free(a_r), tensor_free(b_r), free(final_shape),NULL);
    const int64_t b_r_shape[] = {-1, b_rows, b_cols};
    Tensor *b_f = tensor_reshape(b_r, b_r_shape, 3);
    if (!b_f)
        return (tensor_free(a_r), tensor_free(b_r), tensor_free(a_f), free(final_shape),NULL);
    int64_t batch_size =  tensor_batchsize(a_r->shape, a_r->num_dims);
    const int64_t result_shape[] = {batch_size, a_rows, b_cols};
    Tensor *result = tensor_zeros(result_shape, 3, a->dtype, a->device);
    if (!result)
        return (tensor_free(a_r), tensor_free(b_r), tensor_free(a_f), tensor_free(b_f), free(final_shape),NULL);
    matmul_calculation(a_f, b_f, result);
    final_shape[final_dim - 1] = b_cols;
    final_shape[final_dim - 2] = a_rows;
    Tensor *final_result = tensor_reshape(result, final_shape, final_dim);
    tensor_free(a_f);
    tensor_free(b_f);
    tensor_free_after_reshape(a_r);
    tensor_free_after_reshape(b_r);
    tensor_free_after_reshape(result);
    free(final_shape);
    final_result->is_leaf = 0;
    final_result->grad_fn =  grad_fn;
    if (grad_fn)
        tensor_set_require_grad(final_result, 1);
    return final_result;
}


void matmul_calculation(Tensor *a_r, Tensor *b_r, Tensor *result)
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