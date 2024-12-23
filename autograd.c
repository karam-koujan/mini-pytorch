#include "tensor.h"

Grad_Node	*create_matmul_node(Tensor *a, Tensor *b)
{
	Grad_Node *node;

	node = malloc(sizeof(Grad_Node ));
	// Grad_Node *(**next_functions)(Tensor *,Tensor*);
	// next_functions = malloc(2 * sizeof(	void *(**)(Tensor *,Tensor*)));
	Tensor **saved_tensors = malloc(2 * sizeof(Tensor *));
	if ( !node || !saved_tensors)
	{
		//free(next_functions);
		free(node);
		free(saved_tensors);
		return NULL;
	}
	saved_tensors[0] = a;
	saved_tensors[1] = b;
	// for(int i = 0; i < 2; i++)
	// {
	// 	Tensor *curr = saved_tensors[i];
	// 	if(curr->is_leaf == 1)
	// 	{
	// 		next_functions[i] = tensor_accumulate_grad;
	// 	}else{
	// 		next_functions[i] = curr->grad_fn;
	// 	}
	//}
	//
	// node->next_functions = next_functions;
	node->saved_tensors = saved_tensors;
	node->calculate_gradient = tensor_backmatmul;
	return node;
}
Tensor **tensor_backmatmul(Grad_Node *node, Tensor *grad)
{
	Tensor **res = malloc(2 * sizeof(Tensor *));
	if (!res)
		return NULL;
	Tensor *a = node->saved_tensors[0];
	Tensor *b = node->saved_tensors[1];
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
Grad_Node	*tensor_accumulate_grad(Tensor *a, Tensor *grad)
{
	a->grad = tensor_add(a->grad,grad);
	printf("tensor accumulate");
	return NULL;
}

void	tensor_backward(Tensor *a)
{
	Grad_Node *node = (Grad_Node *)a->grad_fn;
	Tensor *grad_c = tensor_ones(a->num_dims,a->shape,0);
	Tensor **gradients = node->calculate_gradient(node,grad_c);
	Tensor *grad_a = gradients[0];
	Tensor *grad_b = gradients[1];
	tensor_print(grad_a);
	tensor_print(grad_b);
	if (node->saved_tensors[0]->is_leaf == 1)
	{
		tensor_accumulate_grad(node->saved_tensors[0],grad_a);
	}
	else if (node->saved_tensors[0]->is_leaf == 0)
	{
		tensor_backward(node->saved_tensors[0]);
	}
	if (node->saved_tensors[1]->is_leaf == 1)
	{
		tensor_accumulate_grad(node->saved_tensors[1],grad_b);
	}
	else if (node->saved_tensors[1]->is_leaf == 0)
	{
		tensor_backward(node->saved_tensors[1]);
	}

}


int main()
{
	Tensor *a = tensor_full(3,1,2,2,2.0,0);
	Tensor *b = tensor_full(3,1,2,3,3.0,0);
	tensor_set_require_grad(a,1);
	tensor_set_require_grad(b,1);
	Tensor *c = tensor_matmul(a,b);
	// Tensor *c = tensor_pairwise_mul(a,b);
	tensor_backward(c);
	tensor_print(a->grad);
	tensor_print(b->grad);
	
}