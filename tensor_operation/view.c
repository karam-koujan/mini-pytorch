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

static int is_view_allowed(Tensor *a, const int64_t *new_view, int64_t new_ndim)
{
    int new_size = calculate_size(new_view, new_ndim);
    int count = 0;
    if (new_size != a->size)
        return (error_msg("The new tensor should have the same size as the input tensor!"),0);
    for (int i = 0; i < new_ndim; i++)
    {
        if (new_view[i] == -1)
            count++;
    }
    if (count > 1)
        return (error_msg("you should only specifiy 1 -1 to infer shape"), 0);
    return (1);
}

static const int64_t *infer_shape_from_view(Tensor *a, const int64_t *view, int64_t new_ndim)
{
    const   int64_t *new_shape = malloc(new_ndim * sizeof(int64_t));
    if (!new_shape)
        return (error_msg("malloc failed!! in shape creation"), NULL);
    for (int i = 0; i < new_ndim; i++)
    {
        if (view[i] == -1)
            new_shape[i] = size / 
    }
}

Tensor  *tensor_view(Tensor *a, const int64_t *view, int64_t new_ndim)
{
    if (!is_contigious(a))
        return (error_msg("The tensor is not contigious, use contigous or tensor_reshape"), NULL);
    if (!is_view_allowed(a, view, new_ndim))
        return (NULL);
    const   int64_t *new_shape = infer_shape_from_view(a, view, new_ndim);
}
