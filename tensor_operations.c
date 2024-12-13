#include "tensor.h"

int tensor_validate_shape(Tensor *a, Tensor *b)
{

	int a_dim = a->num_dims;
	int b_dim = b->num_dims;
	if (tensor_is_broadcastable(a,b,'m') == -1)
		return -1;
	if(a->shape[a_dim - 1] != b->shape[b_dim - 2])
	{
		fprintf(stderr,"tensor1 and tensor2 shapes cannot be multiplied (%ix%i and %ix%i)\n",a->shape[a_dim - 2],a->shape[a_dim - 1],b->shape[b_dim - 2],b->shape[b_dim - 1]);
		return -1;	
	}

	return 1;
}

int	tensor_is_broadcastable(Tensor *a,Tensor *b, char type)
{
	int	j = type == 'm' ? a->num_dims - 3 : a->num_dims - 1;
	int i = type == 'm' ? b->num_dims - 3 : b->num_dims - 1;
	while (i >= 0 && j >= 0)
	{
		if (a->shape[j] != b->shape[i])
		{                                                                                                              
			if (a->shape[j] != 1 && b->shape[i] != 1)
			{
				fprintf(stderr,"tensor1 and tensor2 are not broadcastable");
				return -1;
			}
		}
		i--;
		j--;
	}
	return 1;
}

float	tensor_get_num(Tensor *a,...)
{
	va_list args;
	va_start(args,a);
	int i = 0;
	int idx = 0;
	while (i < a->num_dims)
	{
		int j = va_arg(args,int);
		idx += j * a->strides[i];
		i++;
	}
	va_end(args);
	float *data = a->data;
	return data[idx];
}

Tensor	**tensor_broadcast(Tensor *a, Tensor *b, char type)
{
	if(tensor_is_broadcastable(a,b,type) == -1)
		return NULL;
	
	int dims = a->num_dims > b->num_dims ? a->num_dims : b->num_dims;

  	int *new_a_shape = malloc(dims * sizeof(int));
    int *new_b_shape = malloc(dims * sizeof(int));
	int *new_a_stride = malloc(dims * sizeof(int));
	int *new_b_stride = malloc(dims * sizeof(int));
	if (!new_a_shape || !new_b_shape || !new_a_stride || !new_b_stride) {
		free(new_a_shape);
		free(new_b_shape);
		free(new_a_stride);
		free(new_b_stride);
		return NULL;
	}

	for (int d = 0; d < dims + 1; d++) {
		new_a_shape[d] = (d >= dims - a->num_dims) ? a->shape[d - (dims - a->num_dims)] : 1;
		new_b_shape[d] = (d >= dims - b->num_dims) ? b->shape[d - (dims - b->num_dims)] : 1;
		new_a_stride[d] = (d >= dims - a->num_dims) ? a->strides[d - (dims - a->num_dims)] : 0;
		new_b_stride[d] = (d >= dims - b->num_dims) ? b->strides[d - (dims - b->num_dims)] : 0;
	}
	for (int i = dims ; i >=0; i--)
	{
		if (new_a_shape[i] == 1 && new_b_shape[i] > 1)
		{
			new_a_shape[i] = new_b_shape[i];
    		new_a_stride[i] = 0;
		}
		else if(new_b_shape[i] == 1 && new_a_shape[i] > 1)
		{
			new_b_shape[i] = new_a_shape[i];
    		new_b_stride[i] = 0;
		}
	
	}
	Tensor *new_a = tensor_empty(1,1,0);
	Tensor *new_b = tensor_empty(1,1,0);
	if (!new_b || !new_a)
	{
		free(new_b);
		free(new_a);
		free(new_a_shape);
		free(new_b_shape);
		free(new_a_stride);
		free(new_b_stride);
		return NULL;
	}
	memcpy(new_a->shape,new_a_shape, dims * sizeof(int));
	memcpy(new_a->strides,new_a_stride, dims * sizeof(int));
	memcpy(new_b->shape,new_b_shape, dims * sizeof(int));
	memcpy(new_b->strides,new_b_stride, dims * sizeof(int));
	new_a->data = a->data;
	new_a->num_dims = dims;
	new_b->data = b->data;
	new_b->num_dims = dims;
	Tensor **res = (Tensor **)malloc(2 * sizeof(Tensor *));
	if (!res)
	{
		free(new_a);
		free(new_b);
		free(new_a_shape);
		free(new_b_shape);
		free(new_a_stride);
		free(new_b_stride);
		return NULL;
	}
	res[0] = new_a;
	res[1] = new_b;

	return res;
}

 
Tensor	*tensor_matmul(Tensor *a, Tensor *b)
{
	if (tensor_validate_shape(a,b) == -1)
		return NULL;
	Tensor **broadcasted_tensors =  tensor_broadcast(a,b,'m');
	if (!broadcasted_tensors)
		return (NULL);
	a = broadcasted_tensors[0];
	b = broadcasted_tensors[1];
	int rows = a->shape[a->num_dims - 2];
	int cols = b->shape[b->num_dims - 1];
	ssize_t	batch_size = tensor_size(a,a->num_dims - 2);
	int	a_shape[3] = {batch_size,rows,a->shape[a->num_dims - 1]};
	int b_shape[3] = {batch_size,b->shape[b->num_dims - 2],cols};
	Tensor *res = tensor_empty(3,batch_size,rows,cols,0);
	Tensor *reshaped_a = tensor_reshape(a,3,a_shape);
	Tensor *reshaped_b = tensor_reshape(b,3,b_shape);

	if (!reshaped_a || !reshaped_b || !res)
	{
		free(reshaped_a);
		free(reshaped_b);
		free(res);
		return NULL;
	}
	for(int b_idx = 0; b_idx < batch_size; b_idx++)
	{
		for(int i = 0; i < rows; i++)
		{
			for(int j = 0; j < cols; j++)
			{
				float sum = 0;
				for(int k = 0; k <  reshaped_a->shape[a->num_dims - 2]; k++)
				{
					sum += tensor_get_num(reshaped_a,b_idx,i,k) * tensor_get_num(reshaped_b,b_idx,k,j);
				}
					float *res_data = res->data;
					int idx = b_idx * res->strides[0] + i * res->strides[1] + j * res->strides[2];
					res_data[idx] = sum;
			}
		}
	}
	free(reshaped_a);
	free(reshaped_b);
	int *res_shape = malloc(a->num_dims * sizeof(int));
	if (!res_shape)
	{
		free(res);
		return NULL;
	}
	memcpy(res_shape,a->shape, a->num_dims * sizeof(int));
	res_shape[a->num_dims - 1] = cols;
	res_shape[a->num_dims - 2] = rows;
	Tensor *result = tensor_reshape(res,a->num_dims,res_shape);
	if (!result)
	{
		free(result);
		free(res_shape);
		free(res);
		return NULL;
	}
	free(res);
	return result;
}

Tensor *tensor_pairwise_operation(Tensor *a, Tensor *b, char operation)
{
	if (tensor_is_broadcastable(a,b,'e') == -1)
		return NULL;
	Tensor **arr = tensor_broadcast(a,b,'e');
	Tensor *res = tensor_empty(1,1,0);
	if (!arr || !res)
	{
		free(arr[0]);
		free(arr[1]);
		free(arr);
		free(res);
		return NULL;
	}
	a = arr[0];
	b = arr[1];
	ssize_t size = tensor_size(a,a->num_dims);
	res->num_dims = a->num_dims;
	memcpy(res->shape,a->shape,a->num_dims * sizeof(int));
	res->data = create_empty_data(a->num_dims,res->shape);
	res->strides = create_stride(a->num_dims,res->shape);
	if (!res->data || !res->data)
	{
		free(arr[0]);
		free(arr[1]);
		free(arr);
		free(res);
		free(res->data);
		free(res->strides);
	}
	int shape[1] = {size};
	Tensor *reshaped_a = tensor_reshape(a,1,shape);
	Tensor *reshaped_b = tensor_reshape(b,1,shape);

	for(int i = 0 ; i < size; i++)
	{
		float *res_data  = res->data;
		if (operation == '+')
			res_data[i] = tensor_get_num(reshaped_a,i) + tensor_get_num(reshaped_b,i);
		if (operation == '-')
			res_data[i] = tensor_get_num(reshaped_a,i) - tensor_get_num(reshaped_b,i);
		if (operation == '*')
			res_data[i] = tensor_get_num(reshaped_a,i) * tensor_get_num(reshaped_b,i);
		if (operation == '/')
			res_data[i] = tensor_get_num(reshaped_a,i) / tensor_get_num(reshaped_b,i);
	}
	return res;
}

Tensor *tensor_add(Tensor *a, Tensor *b)
{
	return tensor_pairwise_operation(a,b,'+');
}
Tensor *tensor_sub(Tensor *a, Tensor *b)
{
	return tensor_pairwise_operation(a,b,'-');
}
Tensor *tensor_div(Tensor *a, Tensor *b)
{
	return tensor_pairwise_operation(a,b,'/');
}
Tensor *tensor_pairwise_mul(Tensor *a, Tensor *b)
{
	return tensor_pairwise_operation(a,b,'*');
}

Tensor *tensor_reshape(Tensor *a,int num_dim,int *shape)
{
	Tensor *res = (Tensor *)malloc(sizeof(Tensor));
	if (!res)
		return NULL;
	res->num_dims = num_dim;
  	res->shape = (int *)malloc(num_dim * sizeof(int));
    if (!res->shape)
    {
        free(res);
        return NULL;
    }
    memcpy(res->shape, shape, num_dim * sizeof(int));
	if (a->strides[0] == 0)
	{
	  res->strides = a->strides;
	}
	else
	{
		res->strides = create_stride(num_dim,shape);
	}
	if (!res->strides)
	{
		free(res);
		return NULL;
	}
	res->data = a->data;
	return res;
}

