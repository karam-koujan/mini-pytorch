#include "../headers/tensor.h"
#include "../headers/print.h"

// Tensor *tensor_reshape(Tensor *a, const int64_t shape)
// {

// }

/*

1. check if the shape has the same size
2. check if the input tensor is contigous
3. create a new shape
4. create a new stride
5. copy the input tensor and add the new stride and shaped
*/

int is_contigious(Tensor *a)
{
    int size = sizeof_type(a->dtype);
    int expected_stride = size;
    for(int i = a->num_dims - 1; i >= 0 ; i--)
    {
        if (expected_stride != a->strides[i])
            return (0);
        expected_stride *= a->shape[i];
    }
    return (1);
}

int is_view_allowed(const int64_t *new_view, int64_t new_ndim)
{
    int count = 0;  
    for (int i = 0; i < new_ndim; i++)
    {
        if (new_view[i] == -1)
            count++;
        if (new_view[i] < -1)
            return (error_msg("the new view can't be bellow -1"),0);
    }
    if (count > 1)
        return (error_msg("you should only specifiy one -1 to infer shape"), 0);
    return (1);
}

int64_t *infer_shape_from_view(Tensor *a, const int64_t *view, int64_t new_ndim)
{
    int64_t *new_shape = malloc(new_ndim * sizeof(int64_t));
    if (!new_shape)
        return (error_msg("malloc failed!! in shape creation"), NULL);
    int64_t view_ele = 1;
    int64_t infered_shape = -1;
    for (int i = 0; i < new_ndim; i++)
    {
        if (view[i] != -1)
            view_ele*=view[i];
        else
            infered_shape = a->size;
    }
    // if (a->size != view_ele && infered_shape != a->size)
    // {
    //  error_msg("The new tensor should have the same size as the input tensor!");
    //  free(new_shape);
    //  return (NULL);
    // }
    infered_shape = tensor_batchsize(a->shape, a->num_dims) / view_ele;
    // if (infered_shape * view_ele != a->size)
    // {
    //     error_msg("The new tensor should have the same size as the input tensor!");
    //     free(new_shape);
    //     return (NULL);      
    // }
    for (int i = 0; i < new_ndim; i++)
    {
        if (view[i] == -1)
            new_shape[i] = infered_shape;
        else
            new_shape[i] = view[i];
    }
    return (new_shape);
}

Tensor  *tensor_view(Tensor *a, const int64_t *view, int64_t new_ndim)
{
    if (!is_contigious(a))
        return (error_msg("The tensor is not contigious, use contigous or tensor_reshape"), NULL);
    if (!is_view_allowed(view, new_ndim))
        return (NULL);
    int64_t *new_shape = infer_shape_from_view(a, view, new_ndim);
    if (!new_shape)
        return (NULL);
    int64_t *new_stride = create_stride(new_shape, new_ndim, a->dtype);
    if (!new_stride)
        return (NULL);
    Tensor *result = malloc(sizeof(Tensor));
    if (!result)
        return (NULL);
    memcpy(result, a, sizeof(Tensor));
    result->num_dims = new_ndim;
    result->shape = new_shape;
    result->strides = new_stride;
    return (result);
}
