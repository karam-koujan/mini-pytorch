#include "tensor.h"
#include <math.h>

Tensor *Linear(Module *module, Tensor *a, int in_features, int out_features, int use_bias)
{
	int *weights_shape = (int *)malloc(a->num_dims * sizeof(int));
	if (!weights_shape)
		return NULL;
	memcpy(weights_shape,a->shape, a->num_dims);
	weights_shape[a->num_dims - 1] = in_features;
	weights_shape[a->num_dims - 2] = out_features;
	Tensor *weights = tensor_ones(a->num_dims, weights_shape,0);
	Tensor *weights_t = tensor_t(weights);
	tensor_set_require_grad(weights_t,1);
	module_param_add(module, weights_t);
	Tensor *result = tensor_matmul(a, weights_t);
	Tensor *bias = bias ? tensor_rand(result->num_dims, result->shape, 0) : NULL;
	if (!bias)
	{
		tensor_set_require_grad(bias,1);
		Tensor *tensor_tmp = result;
		result = tensor_add(result,bias);
		free(tensor_tmp);
	}
	free(weights_shape);
	return result;
}

Tensor *Relu(Tensor *a)
{
	for(int i = 0; i < a->size; i++)
	{
		float *data = a->data;
		data[i] = fmax(data[i],0.0); 
	}
	return a;
}

Module *nn()
{
	Module *module = malloc(sizeof(Module));
	if (!module)
		return NULL;
	module->parameters = NULL;
	return module;
}
int	arr_len(Tensor	**arr)
{
	int i = 0;
	while (arr[i])
		i++;
	return i;
} 


Tensor *foward(Module *module,Tensor *a)
{
	Tensor *layer = Linear(module,a, a->shape[a->num_dims - 1], 5, 0);
	layer = Linear(module,layer, layer->shape[layer->num_dims - 1], 5, 0);
	layer = Linear(module,layer, layer->shape[layer->num_dims - 1], 5, 0);

	return Linear(module,layer, layer->shape[layer->num_dims - 1], 1, 0);
}
Tensor *cost(Tensor *pred, Tensor *label)
{
	Tensor *res = tensor_sub(pred,label);
	Tensor *se  = tensor_pairwise_mul(res,res);
	float	m = 0;
	int i = 0;
	while(i < 4)
	{
		float *data = se->data;
		m+= data[i];
		i++;
	}
	printf("%f",m);
	int mse_shape[2] = {1,1};
	Tensor *mse = tensor_full(2,mse_shape,m / i,0);
	return (mse); 
}

void	module_param_add(Module *module,Tensor *a)
{
	if(!module->parameters)
	{
		module->parameters = malloc(2 * sizeof(Tensor *));
		if (!module->parameters)
			return ;
		module->parameters[0] = malloc(sizeof(Tensor ));
		if (!module->parameters[0])
		{
			free(module->parameters);
			return ;
		}
		memcpy(module->parameters[0], a, sizeof(Tensor));
		module->parameters[1] = NULL;
		return ;
	}
	Tensor **params = module->parameters;
	int	size = arr_len(params);
	Tensor **new_params = malloc((size + 2) * sizeof(Tensor *));
	if (!new_params)
		return ;
	memcpy(new_params, params, size * sizeof(Tensor *));
	new_params[size] = malloc(sizeof(Tensor ));
	memcpy(new_params[size], a, sizeof(Tensor));
	new_params[size + 1] = NULL;
	free(params);
	module->parameters = new_params;
}
void	zero_grad(Module *module)
{
	Tensor **params = module->parameters;
	for(int i = 0;  params[i]; i++)
	{
		Tensor *grad =  params[i]->grad;
		int	*shape = malloc(grad->num_dims * sizeof(int));
		if (!shape)
			return ;
		memcpy(shape, grad->shape, grad->num_dims * sizeof(int));
		Tensor *zero = tensor_zeros(grad->num_dims,shape,0);
		free(grad);
		params[i]->grad = zero;
	}
}
int main()
{
	float data[4][2] = {
	    {0.1, 0.2},
	    {0.3, 0.4},
	    {0.5, 0.6},
	    {0.7, 0.8}
	};
	float labels[4] = {0.3, 0.7, 1.1, 1.5};
	int data_shape[] = {4,2};
	Tensor *d = tensor_tensor(data,data_shape,2);
	int label_shape[] = {4,1};
	Tensor *l = tensor_tensor(labels,label_shape,2);
	Module *module = nn();	
	tensor_print(d);
	tensor_print(l);
	Tensor *prediction;
	for(int epoch = 0; epoch < 50; epoch++)
	{
		prediction = foward(module,d);
	}
	//tensor_print(prediction);
	Tensor *cost_fn = cost(prediction, l);
	tensor_backward(cost_fn, NULL);
	// tensor_print(cost_fn);
	for(int i = 0; module->parameters[i]; i++)
	{
		tensor_print(module->parameters[i]->grad);
	}

}