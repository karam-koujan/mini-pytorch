#include "../headers/tensor.h"
#include "../headers/print.h"

Tensor *tensor_zeros(const int64_t *shape, int64_t ndim, Dtype type, Device device)
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
	tensor->data = create_zero_data(type, tensor->size);
	tensor->device = device;
	tensor->num_dims = ndim;
	tensor->is_leaf = 1;
	tensor->grad_fn = NULL;
	tensor->dtype = type;
	if (!tensor->shape || !tensor->strides)
	{
		error_msg("tensor creation failed!");
		free(tensor);
		free(tensor->shape);
		free(tensor->strides);
	}
	return (tensor);
}

Tensor *tensor_ones(const int64_t *shape, int64_t ndim, Dtype type, Device device)
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
	tensor->data = create_one_data(type, tensor->size);
	tensor->device = device;
	tensor->num_dims = ndim;
	tensor->is_leaf = 1;
	tensor->grad_fn = NULL;
	tensor->dtype = type;
	if (!tensor->shape || !tensor->strides || !tensor->data)
	{
		error_msg("tensor creation failed!");
		free(tensor);
		free(tensor->shape);
		free(tensor->strides);
		free(tensor->data);
	}
	return (tensor);
}

Tensor *tensor_full(const int64_t *shape, int64_t ndim, Dtype type, Device device, void *val)
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
	tensor->data = create_val_data(type, tensor->size, val);
	tensor->device = device;
	tensor->num_dims = ndim;
	tensor->is_leaf = 1;
	tensor->grad_fn = NULL;
	tensor->dtype = type;
	if (!tensor->shape || !tensor->strides || !tensor->data)
	{
		error_msg("tensor creation failed!");
		free(tensor);
		free(tensor->shape);
		free(tensor->strides);
		free(tensor->data);
	}
	return tensor;
}

Tensor *tensor_from_arr(void *arr, const int64_t *shape, int64_t ndim, Dtype type, Device device)
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
	tensor->data = copy_arr_data(arr, type, tensor->size);
	tensor->device = device;
	tensor->num_dims = ndim;
	tensor->is_leaf = 1;
	tensor->grad_fn = NULL;
	tensor->dtype = type;
	if (!tensor->shape || !tensor->strides || !tensor->data)
	{
		error_msg("tensor creation failed!");
		free(tensor);
		free(tensor->shape);
		free(tensor->strides);
		free(tensor->data);
	}
	return tensor;
}

Tensor *tensor_rand(const int64_t *shape, int64_t ndim, Dtype type, Device device)
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
	tensor->data = create_rand_data(type, tensor->size);
	tensor->device = device;
	tensor->num_dims = ndim;
	tensor->is_leaf = 1;
	tensor->grad_fn = NULL;
	tensor->dtype = type;
	if (!tensor->shape || !tensor->strides || !tensor->data)
	{
		error_msg("tensor creation failed!");
		free(tensor);
		free(tensor->shape);
		free(tensor->strides);
		free(tensor->data);
	}
	return tensor;
}

void tensor_infos(Tensor *tensor)
{
	if (!tensor)
		return (error_msg("The tensor is NULL"));
	print_shape(tensor);
	print_strides(tensor);
	printf("size : %i\n", tensor->size);
	printf("requires_grad : %i\n", tensor->requires_grad);
	printf("dims: %i\n", tensor->num_dims);
	printf("is_leaf %i\n", tensor->is_leaf);
	print_device(tensor->device);
	print_type(tensor->dtype);
}

Tensor	*tensor_scalar(void *nb, Dtype type, Device device)
{
	const int64_t shape[] = {1};
	Tensor	*result = tensor_full(shape, 1, type, device, nb);
	return (result);
}
