#include "tensor.h"

typedef struct
{
	Tensor **saved_tensors;
	Tensor	*(**next_functions)(Tensor*,Tensor*);
}	Node;

Node *tensor_backmatmul(Tensor *a, Tensor *b)
{
	Node *node;
	node = malloc(sizeof(Node *));
	Tensor *(**next_functions)(Tensor*,Tensor*);
	next_functions = malloc(2 * sizeof(	Tensor *(*)(Tensor*,Tensor*)));
	Tensor **saved_tensors = malloc(2 * sizeof(Tensor *));
	if (!next_functions || !node || !saved_tensors)
		free(next_functions);
		free(node);
		free(saved_tensors);
		return NULL;
	saved_tensors[0] = a;
	saved_tensors[1] = b;
	for(int i = 0; i < 2; i++)
	{
		Tensor *curr = saved_tensors[i];
		if(curr->is_leaf == 1)
		{
			next_functions[i] = tensor_backmatmul;
		}else{
			next_functions[i] = tensor_backmatmul;
		}
	}
	node->next_functions = next_functions;
	node->saved_tensors = saved_tensors;
	return node; 
}

