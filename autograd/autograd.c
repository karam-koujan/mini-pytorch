#include "../headers/tensor.h"
#include "../headers/print.h"


void tensor_set_require_grad(Tensor *a, int requires_grad)
{
	if (requires_grad == 1)
	{
		int64_t *shape = malloc(a->num_dims * sizeof(int64_t));
		memcpy(shape, a->shape, a->num_dims * sizeof(int64_t));
		a->grad = tensor_zeros(shape, a->num_dims, a->dtype, a->device);
		if (!a->grad)
			return (free(shape), error_msg("a grad is failed in creation"));
		a->requires_grad = 1;
	}
	if (requires_grad == 0)
	{
		tensor_free(a->grad);
		a->grad = NULL;
		a->requires_grad = 0;
	}
}

Tensor *tensor_collapse(Tensor *b_t, Tensor *grad)
{
	int diff = 1;
	int	broadcasted_dim = 1;
	int j = b_t->num_dims - 1;
	for (int i = b_t->prebroadcast_dims - 1; i >= 0; i--)
	{
		if (b_t->shape[j] != b_t->prebroadcast_shape[i])
		{
			diff*= b_t->prebroadcast_shape[i];
			broadcasted_dim*= b_t->shape[j];
		}
		j--;
	}
	while (j>=0)
	{
		broadcasted_dim*= b_t->shape[j];
		j--;
	}
	int64_t coef_int = broadcasted_dim / diff;
	printf("coef int %lli\n", coef_int);
	int64_t	*coef = &coef_int;
	Tensor *co = tensor_full(grad->shape, grad->num_dims, grad->dtype, grad->device, coef);
	Tensor *new_grad = tensor_mul(co, grad);
	Tensor *reduced_grad = tensor_zeros(b_t->prebroadcast_shape, b_t->prebroadcast_dims, b_t->dtype, b_t->device);
	if (!reduced_grad)
	{
		return (tensor_free(co), tensor_free(new_grad),error_msg("an sudden error happens in tensor_mul in tensor_collapse"), grad);
	}
	for (int i = 0; i < calculate_size(b_t->prebroadcast_shape, b_t->prebroadcast_dims); i++)
	{
		fill_data(reduced_grad->data, i, reduced_grad->dtype, (char *)new_grad->data + (i * sizeof_type(reduced_grad->dtype)));
	}	
	if (!new_grad)
	{
		return (tensor_free(co), error_msg("an sudden error happens in tensor_mul in tensor_collapse"), grad);
	}
	tensor_free(new_grad);
	tensor_free(co);
	return reduced_grad;
}


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
	node->broadcasted_tensor_a = NULL;
	node->broadcasted_tensor_b = NULL;
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

Grad_Node	*create_pairwise_mul_node(Tensor *a, Tensor *b)
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
	node->calculate_gradient = tensor_backpairwise_mul;
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
Grad_Node	*create_sub_node(Tensor *a, Tensor *b)
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
	node->calculate_gradient = tensor_backsub;
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
		if (a->is_broadcasted)
		{
			grad_a = tensor_collapse(a, grad);
			tensor_unbroadcast(b);
		}
		else
		{
			grad_a = tensor_deep_copy(grad);
		}
		tensor_set_require_grad(grad_a,0);
	}
	if (b->requires_grad == 1)
	{
		if (b->is_broadcasted)
		{
			grad_b = tensor_collapse(b, grad);
			tensor_unbroadcast(b);
		}
		else
		{
			grad_b = tensor_deep_copy(grad);
		}
		tensor_set_require_grad(grad_b,0);
	
	}
	res[0] = grad_a;
	res[1] = grad_b;
	return res;
}

Tensor **tensor_backsub(Grad_Node *node, Tensor *grad)
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
		if (a->is_broadcasted)
		{
			grad_a = tensor_collapse(a, grad);
			tensor_unbroadcast(b);
		}
		else
		{
			grad_a = tensor_deep_copy(grad);
		}
		tensor_set_require_grad(grad_a,0);
	}
	if (b->requires_grad == 1)
	{
		if (b->is_broadcasted)
		{
			grad_b = tensor_collapse(b, grad);
			tensor_unbroadcast(b);
		}
		else
		{
			grad_b = tensor_deep_copy(grad);
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
	Tensor *b_t = tensor_transpose(b, b->num_dims - 1, b->num_dims - 2);
	Tensor *a_t = tensor_transpose(a, a->num_dims - 1, a->num_dims - 2);

	Tensor *grad_a = NULL;
	Tensor *grad_b = NULL;
	if(a->requires_grad)
	{
		grad_a = tensor_matmul(grad,b_t);
		tensor_set_require_grad(grad_a,0);

	}
	if(b->requires_grad)
	{
		grad_b = tensor_matmul(a_t,grad);
		tensor_set_require_grad(grad_b,0);
	}

	if (node->broadcasted_tensor_a->is_broadcasted)
	{
		
		grad_a = tensor_collapse(node->broadcasted_tensor_a, grad_a);
	}
	if (node->broadcasted_tensor_b->is_broadcasted)
	{
		grad_b = tensor_collapse(node->broadcasted_tensor_b, grad_b);
	}
	res[0] = grad_a;
	res[1] = grad_b;
	free(b_t);
	free(a_t);
	return res;
}

Tensor **tensor_backpairwise_mul(Grad_Node *node, Tensor *grad)
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
			grad_a = grad;
		tensor_set_require_grad(grad_a,0);
	}
	if (b->requires_grad == 1)
	{
		grad_b = grad;
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
	Tensor *new_grad = tensor_add(a->grad,grad);
	if (!new_grad)
		return (error_msg("some thing wrong in tensor_add in tensor accumulate grad"));
	tensor_free(a->grad);
	a->grad = new_grad;
}

void	tensor_backward(Tensor *a, Tensor *prev_grad)
{
	Grad_Node *node = (Grad_Node *)a->grad_fn;

	if (!node)
		return;
	if (!prev_grad)
    {
		prev_grad = tensor_ones(a->shape,a->num_dims,a->dtype, a->device);
        if (!prev_grad)
            return ;
    }
	Tensor **gradients = node->calculate_gradient(node,prev_grad);
	if (!gradients)
		return;
	tensor_free(prev_grad);
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