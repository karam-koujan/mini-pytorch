#include "tensor.h"

Grad_Node	*create_matmul_node(Tensor *a, Tensor *b)
{
	Grad_Node *node;

	node = malloc(sizeof(Grad_Node ));
	Tensor **(**next_functions)(Tensor*,Tensor*,Tensor*);
	next_functions = malloc(2 * sizeof(	void **(*)(Tensor*,Tensor*,Tensor*)));
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
	node->next_functions = next_functions;
	node->saved_tensors = saved_tensors;
	return node;
}
Tensor **tensor_backmatmul(Tensor *a, Tensor *b, Tensor *grad)
{
	Tensor **res = malloc(2 * sizeof(Tensor *));
	if (!res)
		return NULL;
	int batch_size = tensor_size(a,a->num_dims - 2);
	Tensor *b_t = tensor_t(b);
	Tensor *a_t = tensor_t(a);
	Tensor *grad_a = NULL;
	Tensor *grad_b = NULL;
	if(a->requires_grad)
	{
		grad_a = tensor_matmul(grad,b_t);		
		grad_a->requires_grad = 0;
	}
	if(b->requires_grad)
	{
		grad_b = tensor_matmul(a_t,grad);
		grad_b->requires_grad = 0;
	}
	res[0] = grad_a;
	res[1] = grad_b;
	return res;
}
void	tensor_accumulate_grad(Tensor *a, Tensor *grad)
{
	a->grad = tensor_pairwise_add(a,grad);
	printf("tensor accumulate");
}

void	tensor_backward(Tensor *a)
{
	Grad_Node *node = (Grad_Node *)a->grad_fn;
	Tensor *grad_c = tensor_ones(a->num_dims,a->shape,0);
	for(int i = 0; i < 2; i++)
	{
		Tensor **gradients = NULL;
		if (node->next_functions[0])
		{
			gradients = node->next_functions[0](node->saved_tensors[0],node->saved_tensors[1],grad_c);
		}
		if (gradients)
		{
			tensor_print(gradients[0]);
			tensor_print(gradients[1]);
		}
	}
}


int main()
{
	Tensor *a = tensor_full(3,1,2,2,2.0,0);
	a->requires_grad = 1;
	Tensor *b = tensor_full(3,1,2,3,3.0,0);
	b->requires_grad = 1;
	Tensor *c = tensor_matmul(a,b);
	// Tensor *c = tensor_pairwise_mul(a,b);
	tensor_backward(c);

	
}