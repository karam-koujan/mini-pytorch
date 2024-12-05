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
	int i = type == 'm' ? b->num_dims - 3 : a->num_dims - 1;
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

Tensor	*tensor_matmul(Tensor *a, Tensor *b)
{
	if (tensor_validate_shape(a,b) == -1)
		return NULL;
	int rows = a->shape[a->num_dims - 2];
	int cols = b->shape[b->num_dims - 1];
	int	batch_dim = 1;
	for (int i = 0; i < a->num_dims - 2; i++)
	{
		batch_dim*= a->shape[i];
	}
	Tensor *res = tensor_empty(3,batch_dim,rows,cols,NULL,NULL,NULL);
	tensor_reshape(a,3,batch_dim,rows,cols);
	tensor_reshape(b,3,batch_dim,b->shape[b->num_dims - 2],cols);
	for(int b_idx = 0; b_idx < batch_dim; b_idx++)
	{
		for(int i = 0; i < rows; i++)
		{
			for(int j = 0; j < cols; j++)
			{
				float sum = 0;
				for(int k = 0; k < cols; k++)
				{
					sum += tensor_get_num(a,b_idx,i,k) * tensor_get_num(b,b_idx,k,j);
				}
					float *res_data = res->data;
					res_data[b_idx * (rows * cols) + i * cols + j] = sum;
			}
		}
	}

	return res;
}
/*
 Tasks;
 - remove in place reshaping because we lose the shape and strides infos about the tensors
 - reshape the result tensor after calculation
 - add broadcast function
 - test matmul 
 - rewrite tensor creation functions
*/
int main()
{
	tensor_set_seed(1337);
	Tensor a = tensor_rand(4,3,3,2,2,NULL,NULL,NULL);
	Tensor b = tensor_rand(4,3,3,2,2,NULL,NULL,NULL);
	tensor_print(&a);
	tensor_print(&b);
	Tensor *c = tensor_matmul(&a,&b);
	tensor_print(c);
	// Tensor *d = tensor_reshape(&b,3,8,3,2);
	// tensor_print(&b);
}