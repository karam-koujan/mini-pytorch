#include "tensor.h"

typedef struct
{
	Tensor **saved_tensors;
	Tensor	*(**next_functions)(Tensor*,Tensor*);
}	Node;

Tensor	*tensor_backmatmul(Tensor *a, Tensor *b)
{
	Node node;
	Tensor *(**next_functions)(Tensor*,Tensor*);
	next_functions = malloc(2 * sizeof(	Tensor *(*)(Tensor*,Tensor*)));
	Tensor **saved_tensors = malloc(2 * sizeof(Tensor *));
	if (next_functions)
		return NULL;
	saved_tensors[0] = a;
	saved_tensors[1] = b;
	for(i = 0; i < 2; i++)
	{
		Tensor *curr = saved_tensors[i];
		if(a->isleaf == 1)
		{
			next_functions[0] = tensor_backmatmul;
		}else{
			next_functions[0] = tensor_backaccumulate;
		}
	}
}