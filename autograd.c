#include "tensor.h"

void *tensor_backmatmul(Tensor *a, Tensor *b)
{
	Grad_Node *node;

	node = malloc(sizeof(Grad_Node *));
	void *(**next_functions)(Tensor*,Tensor*);
	next_functions = malloc(2 * sizeof(	void *(*)(Tensor*,Tensor*)));
	Tensor **saved_tensors = malloc(2 * sizeof(Tensor *));
	if (!next_functions || !node || !saved_tensors)
	{
		free(next_functions);
		free(node);
		free(saved_tensors);
		return NULL;
	}
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
	Tensor *grad_a = tensor_ones(2,1,1,0);
	tensor_print(grad_a);
	grad_a = tensor_broadcast(grad_a,a,'e')[0];
	grad_a = tensor_pairwise_mul(grad_a,b);
	tensor_print(grad_a);
	node->next_functions = next_functions;
	node->saved_tensors = saved_tensors;
	return node; 
}
void	tensor_accumulate_grad(Tensor *a, Tensor *grad)
{
	a->grad = tensor_add(a->grad,grad);
}

int main()
{
	Tensor *a = tensor_rand(2,2,2,0);
	Tensor *b = tensor_ones(2,2,1,0);
	//Tensor *c = tensor_matmul(a,b);
	Tensor *c = tensor_pairwise_mul(a,b);
	tensor_print(a);
	tensor_print(b);
	tensor_print(c);
}