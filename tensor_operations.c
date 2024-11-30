#include "tensor.h"

int tensor_validate_shape(Tensor a, Tensor b)
{
	if (a.num_dims != b.num_dims)
	{
		fprintf(stderr,"The tensors dim is not equal!!\n");
		return -1;
	}
	int dim = a.num_dims;
	if(a.shape[dim - 1] != b.shape[dim - 2])
	{
		fprintf(stderr,"tensor1 and tensor2 shapes cannot be multiplied (%ix%i and %ix%i)\n",a.shape[dim - 2],a.shape[dim - 1],b.shape[dim - 2],b.shape[dim - 1]);
		return -1;	
	}
	return 1;
}

// Tensor tensor_mul(Tensor a, Tensor b)
// {
	
// }

