#include "../headers/tensor.h"
#include "../headers/print.h"

int is_tensor_broadcastable(Tensor *a, Tensor *b, int mat_p)
{
    int i = mat_p ? a->num_dims - 3 : a->num_dims - 1;
    int j = mat_p ? b->num_dims - 3 :b->num_dims - 1;
    int k = i > j ? j : i;
    while (k >= 0)
    {
        if (a->shape[i] != b->shape[j] && b->shape[j] != 1 && a->shape[i] != 1)
            return (error_msg("the tensors are not broadcastable") ,0);
        i--;
        j--;
        k--;
    }
    return (1);
}

int tensor_broadcast(Tensor *a, Tensor *b)
{
    if (!is_tensor_broadcastable(a, b, 0))
        return (1);
    int ndim = b->num_dims > a->num_dims ? b->num_dims : a->num_dims;
    int64_t *shape_a = (int64_t *)malloc(ndim * sizeof(int64_t));
    int64_t *shape_b = (int64_t *)malloc(ndim * sizeof(int64_t));
    int64_t *stride_a = (int64_t *)malloc(ndim * sizeof(int64_t));
    int64_t *stride_b = (int64_t *)malloc(ndim * sizeof(int64_t));
    if (!shape_a || !shape_b || !stride_a || !stride_b)
        return (error_msg("error in creating shape in tensor_broadcast"), 1);
    int i = ndim - 1;
    int j = a->num_dims - 1;
    int k = b->num_dims - 1; 
    while (i >= 0)
    {
        shape_a[i] = j >= 0 ? a->shape[j] : 1;
        shape_b[i] = k >=0  ? b->shape[k] : 1;
        stride_a[i] = j >= 0 ? a->strides[j] : 0;
        stride_b[i] = k >= 0  ? b->strides[k] : 0;
        i--;
        j--;
        k--;
    }
    i = ndim - 1;
    while (i >= 0)
    {
        if (shape_a[i] != shape_b[i] && shape_a[i] == 1)
        {
            shape_a[i] = shape_b[i];
            stride_a[i] = 0;
            a->is_broadcasted = 1;
        }
        else if (shape_a[i] != shape_b[i] && shape_b[i] == 1)
        {
            shape_b[i] = shape_a[i];
            stride_b[i] = 0;
            b->is_broadcasted = 1;
        }
        i--;
    }
    if (a->is_broadcasted)
    {
        a->prebroadcast_shape = a->shape;
        a->prebroadcast_stride = a->strides;
        a->prebroadcast_dims = a->num_dims;
        a->shape = shape_a;
        a->strides = stride_a;
        a->num_dims = ndim;
    }else
    {
        free(shape_a);
        free(stride_a);
    }
    if (b->is_broadcasted)
    {
        b->prebroadcast_shape = b->shape;
        b->prebroadcast_stride = b->strides;
        b->prebroadcast_dims = b->num_dims;
        b->shape = shape_b;
        b->strides = stride_b;
        b->num_dims = ndim;
    }
    else
    {
        free(shape_b);
        free(stride_b);
    }
    return (0);
}


void tensor_unbroadcast(Tensor *a)
{
    if (!a->is_broadcasted)
        return ;
    free(a->shape);
    free(a->strides);
    a->shape = a->prebroadcast_shape;
    a->strides = a->prebroadcast_stride;
    a->num_dims = a->prebroadcast_dims;
    a->is_broadcasted = 0;
    a->prebroadcast_shape = NULL;
    a->prebroadcast_stride = NULL;
    a->prebroadcast_dims = -1;
}




int tensor_matmul_broadcast(Tensor *a, Tensor *b)
{
    if (!is_tensor_broadcastable(a, b, 1))
        return (1);
    int ndim = b->num_dims > a->num_dims ? b->num_dims : a->num_dims;
    int64_t *shape_a = (int64_t *)malloc(ndim * sizeof(int64_t));
    int64_t *shape_b = (int64_t *)malloc(ndim * sizeof(int64_t));
    int64_t *stride_a = (int64_t *)malloc(ndim * sizeof(int64_t));
    int64_t *stride_b = (int64_t *)malloc(ndim * sizeof(int64_t));
    if (!shape_a || !shape_b || !stride_a || !stride_b)
        return (error_msg("error in creating shape in tensor_broadcast"), 1);
    int i = ndim - 1;
    int j = a->num_dims - 1;
    int k = b->num_dims - 1; 
    while (i >= 0)
    {
        shape_a[i] = j >= 0 ? a->shape[j] : 1;
        shape_b[i] = k >=0  ? b->shape[k] : 1;
        stride_a[i] = j >= 0 ? a->strides[j] : 0;
        stride_b[i] = k >= 0  ? b->strides[k] : 0;
        i--;
        j--;
        k--;
    }
    i = ndim - 3;
    while (i >= 0)
    {
        if (shape_a[i] != shape_b[i] && shape_a[i] == 1)
        {
            shape_a[i] = shape_b[i];
            stride_a[i] = 0;
            a->is_broadcasted = 1;
        }
        else if (shape_a[i] != shape_b[i] && shape_b[i] == 1)
        {
            shape_b[i] = shape_a[i];
            stride_b[i] = 0;
            b->is_broadcasted = 1;
        }
        i--;
    }
    if (a->is_broadcasted)
    {
        a->prebroadcast_shape = a->shape;
        a->prebroadcast_stride = a->strides;
        a->prebroadcast_dims = a->num_dims;
        a->shape = shape_a;
        a->strides = stride_a;
        a->num_dims = ndim;
    }else
    {
        free(shape_a);
        free(stride_a);
    }
    if (b->is_broadcasted)
    {
        b->prebroadcast_shape = b->shape;
        b->prebroadcast_stride = b->strides;
        b->prebroadcast_dims = b->num_dims;
        b->shape = shape_b;
        b->strides = stride_b;
        b->num_dims = ndim;
    }
    else
    {
        free(shape_b);
        free(stride_b);
    }
    return (0);
}


int64_t tensor_size(Tensor *a)
{
    int64_t s = 1;
    int p = 0;
    for(int64_t i = 0 ; i < a->num_dims ; i++)
    {
        p = 1;
        s*= a->shape[i];
    }
    return p == 0 ? 0 : s;
}


int tensor_contigous_broadcast(Tensor *a)
{
    if (!a->is_broadcasted)
        return (1);
    int64_t *new_stride = create_stride(a->shape, a->num_dims, a->dtype);
    if (!new_stride)
        return (1);

    int64_t size = tensor_size(a);
    void *data = calloc(size, sizeof_type(a->dtype));
    if (!data)
        return (1);
    int64_t b_t = size / a->size;
    printf("size :%d", a->size);
    for (int64_t i = 0; i < b_t ; i++)
    {
        for(int j = 0; j < a->size; j++)
        {
            int idx = i * a->size + j ;
fill_data(data, idx, a->dtype, (char*)a->data + j * sizeof_type(a->dtype));
        }
    }
    tensor_print(a);
    free(a->data);
    free(a->strides);
    a->size = size;
    a->strides = new_stride;
    a->data = data;
    printf("I am here tesitning tenso contigious \n");
    tensor_print(a);
    tensor_infos(a);
    return (0);
}