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
 	while(i >= 0 && j < 0)
	{
		if(b->shape[i] != 1)
		{
			fprintf(stderr,"tensor1 and tensor2 are not broadcastable");
			return -1;
		}
		i--;
	}
	while(j >= 0 && i < 0)
	{
		if(a->shape[j] != 1)
		{
			fprintf(stderr,"tensor1 and tensor2 are not broadcastable");
			return -1;
		}
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
	
	int	j = type == 'm' ? a->num_dims - 3 : a->num_dims - 1;
	int i = type == 'm' ? b->num_dims - 3 : b->num_dims - 1;
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

	for (int d = 0; d < dims; d++) {
		new_a_shape[d] = (d >= dims - a->num_dims) ? a->shape[d - (dims - a->num_dims)] : 1;
		new_b_shape[d] = (d >= dims - b->num_dims) ? b->shape[d - (dims - b->num_dims)] : 1;
		new_a_stride[d] = (d >= dims - a->num_dims) ? a->strides[d - (dims - a->num_dims)] : 0;
		new_b_stride[d] = (d >= dims - b->num_dims) ? b->strides[d - (dims - b->num_dims)] : 0;
	}
    while (i >= 0 && j >= 0) {
        if (a->shape[j] > b->shape[i]) {
            new_b_shape[i] = a->shape[j];
        }
        if (a->shape[j] < b->shape[i]) {
        	new_a_shape[i] = b->shape[j];
        }
        i--;
        j--;
    }
	while(i >= 0 && j < 0)
	{
    new_a_shape[i] = b->shape[i];
    new_a_stride[i] = 0;
    i--;
	}
	while(j >= 0 && i < 0)
	{
		new_b_shape[j] = a->shape[j];
		new_b_stride[j] = 0;
		j--;
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
	new_a->shape = new_a_shape;
	new_a->data = a->data;
	new_a->strides = new_a_stride;
	new_a->num_dims = dims;
	new_b->shape = new_b_shape;
	new_b->data = b->data;
	new_b->strides = new_b_stride;
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
	Tensor **broadcasted_tensors = tensor_broadcast(a,b,'m');
	if (!broadcasted_tensors)
		return (NULL);
	a = broadcasted_tensors[0];
	b = broadcasted_tensors[1];
	int rows = a->shape[a->num_dims - 2];
	int cols = b->shape[b->num_dims - 1];
	int	batch_size = 1;
	for (int i = 0; i < a->num_dims - 2; i++)
	{
		batch_size*= a->shape[i];
	}
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

Tensor *tensor_reshape(Tensor *a,int num_dim,int *shape)
{
	Tensor *res = (Tensor *)malloc(sizeof(Tensor));
	if (!res)
		return NULL;
	res->num_dims = num_dim;
	res->shape = shape;
	res->strides = create_stride(num_dim, shape);
	if (!res->strides)
	{
		free(res);
		return NULL;
	}
	res->data = a->data;
	return res;
}
/*
 Tasks;
 - create stride and remove inplace
 - test matmul 
 - rewrite tensor creation functions
*/
void f()
{
	system("leaks a.out");
}
int main()
{
	tensor_set_seed(1337);
	Tensor *a = tensor_rand(3,2,2,3,0);
	Tensor *b = tensor_rand(4,1,2,3,5,0);


	//Tensor **arr = tensor_broadcast(a,b,'m');
	// tensor_print(a);
	// //tensor_print(b);
	// int	a_shape[3] = {2,2,3};
	// int b_shape[3] = {2,3,5};
	// Tensor *reshaped_a = tensor_reshape(a,3,a_shape);
	// Tensor *reshaped_b = tensor_reshape(b,3,b_shape);
	// for(int b_idx = 0; b_idx < 2; b_idx++)
	// {
	// 	for(int i = 0; i < 2; i++)
	// 	{
	// 		for(int j = 0; j < 5; j++)
	// 		{
	// 			for(int k = 0; k < 2; k++)
	// 			{
	// 				printf("element: %f  idx : %i\n",tensor_get_num(reshaped_b,b_idx,k,j),b_idx);
	// 			}
		
	// 		}
	// 	}
	// }
	// free(c);
	// free(a);
	// free(b);
	// atexit(f);
	 Tensor *c = tensor_matmul(a,b);
	 tensor_print(c);
	// tensor_print(a);
	// printf("Tensor num %f",tensor_get_num(a,2,2,1,2));
	// Tensor *d = tensor_reshape(&b,3,8,3,2);
	// tensor_print(&b);
}