#include "tensor.h"



Tensor *add_grad(Tensor *a, Tensor *grad)
{
    // Extract dimensions
    int cols = grad->shape[grad->num_dims - 1]; // cols of B
    int rows = grad->shape[grad->num_dims - 2]; // rows of grad
    int common_dim = a->shape[a->num_dims - 1]; // common_dim between A and B
    ssize_t batch_size = tensor_size(a, a->num_dims - 3); // Batch size

    // Reshape tensors for easier manipulation
    int a_shape[3] = {batch_size, rows, common_dim};
    int grad_shape[3] = {batch_size, rows, cols};
    Tensor *reshaped_a = tensor_reshape(a, 3, a_shape);
    Tensor *reshaped_grad = tensor_reshape(grad, 3, grad_shape);

    // Access raw data
    float *grad_data = grad->data;

    // Compute gradients
    for (int b_idx = 0; b_idx < batch_size; b_idx++) {
        for (int i = 0; i < common_dim; i++) { // Iterate over rows of B
            for (int k = 0; k < cols; k++) {   // Iterate over cols of B
                float sum = 0;
                for (int j = 0; j < rows; j++) { // Iterate over rows of A and grad
                    float a_val = tensor_get_num(reshaped_a, b_idx, j, i);
                    float grad_val = tensor_get_num(reshaped_grad, b_idx, j, k);
                    sum += a_val * grad_val;
                }
                // Compute index for grad_data
                int idx = b_idx * grad->strides[0] + i * grad->strides[1] + k * grad->strides[2];
                grad_data[idx] = sum;
            }
        }
    }

    return grad;
}


Tensor *calculate_grad(Tensor *a, Tensor *b)
{
	Tensor *grad_a = tensor_ones(a->num_dims,a->shape,0);
	Tensor *grad_b = tensor_ones(b->num_dims,b->shape,0);
	
	
	if (grad_a->size > grad_b->size)
	{
		grad_a = tensor_pairwise_mul(grad_a,b);
		grad_b = add_grad(a,grad_b);
	}
	return grad_b;
}
Grad_Node	*create_matmul_node(Tensor *a, Tensor *b)
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
	node->next_functions = next_functions;
	node->saved_tensors = saved_tensors;
	return node;
}
Tensor **tensor_backmatmul(Tensor *a, Tensor *b)
{
	Tensor **res = malloc(2 * sizeof(Tensor *));
	if (!res)
		return NULL;
	int batch_size = tensor_size(a,a->num_dims - 2);
	int shape[3] = {batch_size,a->shape[a->num_dims - 2],b->shape[b->num_dims - 1]};
	Tensor *grad_c = tensor_ones(a->num_dims,shape,0);
	Tensor *b_t = tensor_t(b);
	Tensor *a_t = tensor_t(a);
	
	Tensor *grad_a = tensor_matmul(grad_c,b_t);
	Tensor *grad_b = tensor_matmul(a_t,grad_c);
	res[0] = grad_a;
	res[1] = grad_b;
	return res;
}
void	tensor_accumulate_grad(Tensor *a, Tensor *grad)
{
	a->grad = tensor_add(a->grad,grad);
}

void	tensor_backward(Tensor *a)
{

}

int main()
{
	Tensor *a = tensor_full(3,1,2,2,2.0,0);
	Tensor *b = tensor_full(3,1,2,3,3.0,0);
	//Tensor *c = tensor_matmul(a,b);
	// Tensor *c = tensor_pairwise_mul(a,b);
	tensor_backmatmul(a,b);

	
}