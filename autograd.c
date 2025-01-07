#include "tensor.h"

Grad_Node	*create_matmul_node(Tensor *a, Tensor *b)
{
	Grad_Node *node;
	node = malloc(sizeof(Grad_Node ));
	Tensor **saved_tensors = malloc(2 * sizeof(Tensor *));
	if ( !node || !saved_tensors)
	{
		free(node);
		free(saved_tensors);
		return NULL;
	}

	saved_tensors[0] = a;
	saved_tensors[1] = b;
	node->saved_tensors = saved_tensors;
	node->calculate_gradient = tensor_backmatmul;
	return node;
}

Grad_Node	*create_mm_node(Tensor *a, Tensor *b)
{
	Grad_Node *node;

	node = malloc(sizeof(Grad_Node ));
	Tensor **saved_tensors = malloc(2 * sizeof(Tensor *));
	if ( !node || !saved_tensors)
	{
		free(node);
		free(saved_tensors);
		return NULL;
	}
	saved_tensors[0] = a;
	saved_tensors[1] = b;
	node->saved_tensors = saved_tensors;
	node->calculate_gradient = tensor_backmm;
	return node;
}

Grad_Node	*create_add_node(Tensor *a, Tensor *b)
{
	Grad_Node *node;

	node = malloc(sizeof(Grad_Node ));
	Tensor **saved_tensors = malloc(2 * sizeof(Tensor *));
	if ( !node || !saved_tensors)
	{
		free(node);
		free(saved_tensors);
		return NULL;
	}
	saved_tensors[0] = a;
	saved_tensors[1] = b;
	node->saved_tensors = saved_tensors;
	node->calculate_gradient = tensor_backadd;
	return node;
}

Tensor **tensor_backadd(Grad_Node *node, Tensor *grad)
{
	Tensor *a = node->saved_tensors[0];
	Tensor *b = node->saved_tensors[1];
	Tensor **res = malloc(2 * sizeof(Tensor *));
	if (!res)
		return NULL;
	Tensor *grad_a = NULL;
	Tensor *grad_b = NULL;
	if (a->requires_grad == 1)
	{
		grad_a = tensor_collapse(grad,a->shape,a->num_dims);
		if (!grad_a)
		{
			grad_a = grad;
		}
		tensor_set_require_grad(grad_a,0);
	}
	if (b->requires_grad == 1)
	{
		grad_b = tensor_collapse(grad,b->shape,b->num_dims);
		if (!grad_b)
		{
			grad_b = grad;
		}
		tensor_set_require_grad(grad_b,0);
	}
	res[0] = grad_a;
	res[1] = grad_b;
	return res;
}


Tensor **tensor_backmatmul(Grad_Node *node, Tensor *grad)
{
	Tensor **res = malloc(2 * sizeof(Tensor *));
	if (!res)
		return NULL;
	Tensor *a = node->saved_tensors[0];
	Tensor *b = node->saved_tensors[1];
	Tensor *b_t = tensor_t(b);
	Tensor *a_t = tensor_t(a);
	Tensor *grad_a = NULL;
	Tensor *grad_b = NULL;
	if(a->requires_grad)
	{

		grad_a = tensor_matmul(grad,b_t);
		if (!grad_a)
			return NULL;
		grad_a = tensor_collapse(grad_a,a->shape,a->num_dims);
		if (!grad_a)
		{
			grad_a = tensor_matmul(grad,b_t);
		}
		tensor_set_require_grad(grad_a,0);
	}
	if(b->requires_grad)
	{
		grad_b = tensor_matmul(a_t,grad);
		if (!grad_b)
			return NULL;
		grad_b =tensor_collapse(grad_b,b->shape,b->num_dims);
		if (!grad_b)
		{
			grad_b = tensor_matmul(a_t,grad);
		}
		tensor_set_require_grad(grad_b,0);
	}
	res[0] = grad_a;
	res[1] = grad_b;
	return res;
}
Tensor **tensor_backmm(Grad_Node *node, Tensor *grad)
{
	Tensor **res = malloc(2 * sizeof(Tensor *));
	if (!res)
		return NULL;
	Tensor *a = node->saved_tensors[0];
	Tensor *b = node->saved_tensors[1];
	Tensor *b_t = tensor_t(b);
	Tensor *a_t = tensor_t(a);
	Tensor *grad_a = NULL;
	Tensor *grad_b = NULL;
	if(a->requires_grad)
	{
		grad_a = tensor_mm(grad,b_t);		
		tensor_set_require_grad(grad_a,0);
	}
	if(b->requires_grad)
	{
		grad_b = tensor_mm(a_t,grad);
		tensor_set_require_grad(grad_b,0);
	}
	res[0] = grad_a;
	res[1] = grad_b;
	return res;
}
void	tensor_accumulate_grad(Tensor *a, Tensor *grad)
{
	a->grad = tensor_add(a->grad,grad);
}

void	tensor_backward(Tensor *a, Tensor *prev_grad)
{
	Grad_Node *node = (Grad_Node *)a->grad_fn;
	if (!node)
		return;
	if (!prev_grad)
		prev_grad = tensor_ones(a->num_dims,a->shape,0);
	Tensor **gradients = node->calculate_gradient(node,prev_grad);
	if (!gradients)
		return;
	Tensor *grad_a = gradients[0];
	Tensor *grad_b = gradients[1];
	if (node->saved_tensors[0]->is_leaf == 1 && node->saved_tensors[0]->requires_grad == 1)
	{
		tensor_accumulate_grad(node->saved_tensors[0],grad_a);
	}
	else if (node->saved_tensors[0]->is_leaf == 0 && node->saved_tensors[0]->requires_grad == 1)
	{
		tensor_backward(node->saved_tensors[0],grad_a);
	}
	if (node->saved_tensors[1]->is_leaf == 1 && node->saved_tensors[1]->requires_grad == 1)
	{
		tensor_accumulate_grad(node->saved_tensors[1],grad_b);
	}
	else if (node->saved_tensors[1]->is_leaf == 0 && node->saved_tensors[1]->requires_grad == 1)
	{
		tensor_backward(node->saved_tensors[1],grad_b);
	}

}

