#include "../headers/tensor.h"
#include "../headers/print.h"

Tensor *tensor_zeros(const int64_t *shape, int64_t ndim, Dtype type, Device device)
{
	Tensor *tensor = tensor_constructor(shape, ndim, type, device);
	if (!tensor)
		return (NULL);
	tensor->data = create_zero_data(type, tensor->size);
	if (!tensor->shape || !tensor->strides)
	{
		error_msg("tensor creation failed!");
		free(tensor->shape);
		free(tensor->strides);
		free(tensor);
	}
	return (tensor);
}

Tensor *tensor_ones(const int64_t *shape, int64_t ndim, Dtype type, Device device)
{
	Tensor *tensor = tensor_constructor(shape, ndim, type, device);
	if (!tensor)
		return (NULL);
	tensor->data = create_one_data(type, tensor->size);
	if (!tensor->shape || !tensor->strides || !tensor->data)
	{
		error_msg("tensor creation failed!");
		free(tensor->shape);
		free(tensor->strides);
		free(tensor->data);
		free(tensor);
	}
	return (tensor);
}

Tensor *tensor_full(const int64_t *shape, int64_t ndim, Dtype type, Device device, void *val)
{
	Tensor *tensor = tensor_constructor(shape, ndim, type, device);
	if (!tensor)
		return (NULL);
	tensor->data = create_val_data(type, tensor->size, val);
	if (!tensor->shape || !tensor->strides || !tensor->data)
	{
		error_msg("tensor creation failed!");
		free(tensor->shape);
		free(tensor->strides);
		free(tensor->data);
		free(tensor);
	}
	return tensor;
}

Tensor *tensor_from_arr(void *arr, const int64_t *shape, int64_t ndim, Dtype type, Device device)
{
	Tensor *tensor = tensor_constructor(shape, ndim, type, device);
	if (!tensor)
		return (NULL);
	tensor->data = copy_arr_data(arr, type, tensor->size);

	if (!tensor->shape || !tensor->strides || !tensor->data)
	{
		error_msg("tensor creation failed!");
		free(tensor->shape);
		free(tensor->strides);
		free(tensor->data);
		free(tensor);
	}
	return tensor;
}

Tensor *tensor_rand(const int64_t *shape, int64_t ndim, Dtype type, Device device)
{
	Tensor *tensor = tensor_constructor(shape, ndim, type, device);
	if (!tensor)
		return (NULL);
	tensor->data = create_rand_data(type, tensor->size);
	if (!tensor->shape || !tensor->strides || !tensor->data)
	{
		error_msg("tensor creation failed!");
		free(tensor->shape);
		free(tensor->strides);
		free(tensor->data);
		free(tensor);
	}
	return tensor;
}

void tensor_infos(Tensor *tensor)
{
	if (!tensor)
		return (error_msg("The tensor is NULL"));
	print_shape(tensor->shape, tensor->num_dims);
	print_strides(tensor->strides, tensor->num_dims);
	printf("size : %i\n", tensor->size);
	printf("requires_grad : %i\n", tensor->requires_grad);
	printf("dims: %i\n", tensor->num_dims);
	printf("is_leaf %i\n", tensor->is_leaf);
	print_device(tensor->device);
	print_type(tensor->dtype);
	if (tensor->is_broadcasted)
	{
		printf("broadcasted tensor metadata :\n");
		print_shape(tensor->prebroadcast_shape, tensor->prebroadcast_dims);
		print_strides(tensor->prebroadcast_stride, tensor->prebroadcast_dims);

	}
}

Tensor	*tensor_scalar(void *nb, Dtype type, Device device)
{
	const int64_t shape[] = {1};
	Tensor	*result = tensor_full(shape, 1, type, device, nb);
	return (result);
}
