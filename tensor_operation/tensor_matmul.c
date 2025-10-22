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
    
    if (a->num_dims == 2 && b->num_dims == 2) {
        return tensor_mm(a, b);
    }

	Grad_Node *grad_fn = a->requires_grad || b->requires_grad ? create_matmul_node(a, b) : NULL;

    Tensor *a_view = tensor_copy(a);
    if (!a_view) return NULL;
    Tensor *b_view = tensor_copy(b);
    if (!b_view) {
        free(a_view);
        return NULL;
    }


    if (a->num_dims >= 3 || b->num_dims >= 3) {
        if (tensor_matmul_broadcast(a_view, b_view)) {
            tensor_free_view(a_view);
            tensor_free_view(b_view);
            return NULL;
        }
        grad_fn->broadcasted_shape_a = create_shape(a_view->shape, a->num_dims);
        if (!grad_fn->broadcasted_shape_a)
        {
            tensor_free_view(a_view);
            tensor_free_view(b_view);
            return NULL;      
        }
        grad_fn->broadcasted_shape_b = create_shape(b_view->shape, b->num_dims);
        if (!grad_fn->broadcasted_shape_b)
        {
            free(grad_fn->broadcasted_shape_a);
            tensor_free_view(a_view);
            tensor_free_view(b_view);
            return NULL;        
        }
    }
    

    const int64_t a_view_shape[] = {-1, a_view->shape[a_view->num_dims - 2], a_view->shape[a_view->num_dims - 1]};
    Tensor *a_flat = tensor_reshape(a_view, a_view_shape, 3);
    
    const int64_t b_view_shape[] = {-1, b_view->shape[b_view->num_dims - 2], b_view->shape[b_view->num_dims - 1]};
    Tensor *b_flat = tensor_reshape(b_view, b_view_shape, 3);

    int final_dim = a_view->num_dims;
    int64_t *final_shape = create_shape(a_view->shape, final_dim);
    if (!final_shape) {
        tensor_free_view(a_view);
        tensor_free_view(b_view);
        return (NULL);
    }
    final_shape[final_dim - 2] = a_view->shape[a_view->num_dims - 2];
    final_shape[final_dim - 1] = b_view->shape[b_view->num_dims - 1];
    
    const int64_t result_shape[] = {a_flat->shape[0], a_flat->shape[1], b_flat->shape[2]};
    Tensor *result_flat = tensor_zeros(result_shape, 3, a->dtype, a->device);
    
    if (!a_flat || !b_flat || !result_flat) {
        error_msg("Failed to allocate intermediate tensors for matmul.");
        tensor_free_view(a_view);
        tensor_free_view(b_view);
        tensor_free_after_reshape(a_flat);
        tensor_free_after_reshape(b_flat);
        tensor_free(result_flat);
        free(final_shape);
        return NULL;
    }
    
    matmul_calculation(a_flat, b_flat, result_flat);
    
    Tensor *final_result = tensor_reshape(result_flat, final_shape, final_dim);
    final_result->is_leaf = 0;
    final_result->grad_fn = grad_fn;

    tensor_free_view(a_view);
    tensor_free_view(b_view);
    tensor_free_after_reshape(a_flat);
    tensor_free_after_reshape(b_flat);
    tensor_free_after_reshape(result_flat);
    free(final_shape);

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