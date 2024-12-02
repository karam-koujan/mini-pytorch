#include "tensor.h"

int tensor_validate_shape(Tensor *a, Tensor *b)
{
	if (a->num_dims != b->num_dims)
	{
		fprintf(stderr,"The tensors dim is not equal!!\n");
		return -1;
	}
	int dim = a->num_dims;
	if(a->shape[dim - 1] != b->shape[dim - 2])
	{
		fprintf(stderr,"tensor1 and tensor2 shapes cannot be multiplied (%ix%i and %ix%i)\n",a->shape[dim - 2],a->shape[dim - 1],b->shape[dim - 2],b->shape[dim - 1]);
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

// Tensor *tensor_matmul(Tensor *a, Tensor *b)
// {
	
// 	if (tensor_validate_shape(a,b) == -1 || tensor_is_broadcastable(a,b,'m') == -1)
// 		return NULL;
// 	Tensor tensors[2] = broadcast_tensor(a , b);
// 	a = tensors;
// 	b = tensors + 1;
// }
