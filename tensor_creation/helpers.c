#include "../headers/tensor.h"
#include "../headers/print.h"

int sizeof_type(Dtype type)
{
    int size = -1;
    if (type == FLOAT32 || type == INT32)
        size = 4;
    else if (type == DOUBLE || type == INT64)
        size = 8;
    else
        return (error_msg("invalid type!"), size);
    return (size);
}

int64_t calculate_size(const int64_t *shape, int64_t ndim)
{
    int size = 1;
    for (int i = 0; i < ndim; i++)
    {
        size*= shape[i];
    }
    return (size);
}

void tensor_set_seed(unsigned int seed)
{
	srand(seed);
}

float	generate_random()
{
     float r = (float)rand() / (float)RAND_MAX;
    
	return r;	
}

int64_t *create_shape(const int64_t *shape, int64_t ndim)
{
    int64_t *new_shape = (int64_t *)malloc(ndim  * sizeof(int64_t));
    if (!new_shape)
        return (error_msg("shape creation failed!"), NULL);
    memcpy(new_shape, shape, sizeof(int64_t) * ndim);
    return (new_shape);
}

int64_t *create_stride(const int64_t *shape, int64_t ndim, Dtype type)
{
    int size = sizeof_type(type);
    if (size == -1)
        return (NULL);
    int64_t *strides = (int64_t *)malloc(ndim  * sizeof(int64_t));
    if (!strides)
        return (error_msg("stride creation failed!"), NULL);
    strides[ndim - 1] = size;
    for (int i = ndim - 2 ; i >= 0; i--)
    {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return (strides);
}

Tensor *tensor_deep_copy(Tensor *a)
{
    if (a == NULL)
    {
        error_msg("you entred an empty tensor");
        return (NULL);
    }
    Tensor *r = malloc(sizeof(Tensor));
    if (!r)
        return (NULL);
    memcpy(r, a, sizeof(Tensor));

    void *data = malloc(a->size * sizeof_type(a->dtype));
    r->grad_fn = NULL;
    if (!data)
    {
        r->prebroadcast_shape = NULL;
        r->prebroadcast_stride = NULL;
        r->data = NULL;
        r->shape = NULL;
        r->strides = NULL;
        r->grad = NULL;
        return (tensor_free(r), NULL);
    }
    memcpy(data, a->data, a->size * sizeof_type(a->dtype));
    r->data = data;
    int64_t *shape = malloc(a->num_dims * sizeof(int64_t));
    if (!shape)
    {
        r->prebroadcast_shape = NULL;
        r->prebroadcast_stride = NULL;
        r->shape = NULL;
        r->strides = NULL;
        r->grad = NULL;
        return (tensor_free(r), NULL);
    }
    memcpy(shape, a->shape, a->num_dims * sizeof(int64_t));
    r->shape = shape;
    int64_t *strides = malloc(a->num_dims * sizeof(int64_t));
    if (!strides)
    {
        r->prebroadcast_shape = NULL;
        r->prebroadcast_stride = NULL;
        r->strides = NULL;
        r->grad = NULL;
        return (tensor_free(r), NULL);
    }
    r->strides = strides;
    memcpy(strides, a->strides, a->num_dims * sizeof(int64_t));
    Tensor  *grad = NULL;
    if (a->grad)
    {
        grad = tensor_deep_copy(a->grad);
        if (!grad)
        {
            r->prebroadcast_shape = NULL;
            r->prebroadcast_stride = NULL;
            r->grad = NULL;
            return (tensor_free(r), NULL);
        }
    }
    r->grad = NULL;
    if (a->is_broadcasted)
    {
        r->prebroadcast_shape = malloc(a->prebroadcast_dims * sizeof(int64_t));
        r->prebroadcast_stride = malloc(a->prebroadcast_dims * sizeof(int64_t));
        if (!r->prebroadcast_shape || !r->prebroadcast_stride)
        {
            r->prebroadcast_shape = NULL;
            r->prebroadcast_stride = NULL;
            return (tensor_free(r), NULL);
        }
        memcpy(r->prebroadcast_shape, a->prebroadcast_shape, a->prebroadcast_dims * sizeof(int64_t));
        memcpy(r->prebroadcast_stride, a->prebroadcast_stride, a->prebroadcast_dims * sizeof(int64_t));
    }
    r->grad_fn = NULL;
    return (r);
}

Tensor *tensor_constructor(const int64_t *shape, int ndim, Dtype type, Device device)
{
	if (shape == NULL)
		return (error_msg("invalid shape"), NULL);
	if (ndim <= 0)
		return (error_msg("invalid ndim"), NULL);
	Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));
	if (!tensor)
		return (error_msg("tensor creation failed!"), NULL);
	tensor->shape = create_shape(shape, ndim);
	tensor->strides = create_stride(shape, ndim, type);
	tensor->size = calculate_size(shape, ndim);
	tensor->device = device;
	tensor->num_dims = ndim;
	tensor->is_leaf = 1;
	tensor->grad_fn = NULL;
	tensor->dtype = type;
	tensor->prebroadcast_dims = -1;
	tensor->prebroadcast_shape = NULL;
	tensor->prebroadcast_stride = NULL;
	tensor->is_broadcasted = 0;
	tensor->grad = NULL;
	tensor->requires_grad = 0;
    return tensor;
}




double rand_uniform(double a, double b) {
    // The standard, safe way to get a random double between 0.0 and 1.0
    // Explicitly cast to double BEFORE division to avoid integer arithmetic.
    double u = (double)rand() / (double)RAND_MAX;
    
    // Scale and shift to the desired range [a, b]
    return a + u * (b - a);
}