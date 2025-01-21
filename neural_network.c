#include "tensor.h"
#include "plot_loss.h"
#include <math.h>

int	arr_len(Tensor	**arr)
{
	if (arr == NULL)
		return 0;
	int i = 0;
	while (arr[i])
		i++;
	return i;
} 

Tensor *Linear(Module *module, int layernum, Tensor *a, int in_features, int out_features, int use_bias)
{
	int *weights_shape = (int *)malloc(3 * sizeof(int));
	if (!weights_shape)
		return NULL;

	weights_shape[0] = 1;
	weights_shape[a->num_dims - 1] = in_features;
	weights_shape[a->num_dims - 2] = out_features;
	int params_len = arr_len(module->parameters);
	Tensor *weights = tensor_rand(a->num_dims, weights_shape,0);
	Tensor *weights_t = tensor_t(weights);
	if (layernum < params_len)
	{
		weights_t = module->parameters[layernum];
		weights_t->is_leaf = 1;
		tensor_set_require_grad(weights_t,1);
	}else
	{
		tensor_set_require_grad(weights_t,1);
		module_param_add(module, weights_t);
	}
	Tensor *result = tensor_matmul(a, weights_t);
	Tensor *bias = use_bias ? tensor_rand(result->num_dims, result->shape, 0) : NULL;
	if (module->parameters[layernum] == NULL)
		
	if (bias)
	{
		tensor_set_require_grad(bias,1);
		Tensor *tensor_tmp = result;
		result = tensor_add(result,bias);
		free(tensor_tmp);
	}
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
		module->parameters[0] =  a;
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


Tensor *mse(Tensor *pred, Tensor *label)
{
	label->is_leaf = 0;
	Tensor *res = tensor_sub(pred,label);
	Tensor *se  = tensor_pairwise_mul(res,res);
	int  one_d_shape[1] = {1};
	Tensor	*m = tensor_zeros(1,one_d_shape,0);
	tensor_set_require_grad(m,1);
	int i = 0;
	while(i < pred->shape[pred->num_dims - 2])
	{
		float *data = se->data;
		Tensor *data_t = tensor_full(1,one_d_shape,data[i],0);
		Tensor *m_c = m;
		m = tensor_add(m,data_t);
		free(m_c);
		i++;
	}
	int mse_shape[3] = {1,1,1};
	Tensor *div = tensor_full(1,one_d_shape,1.0 / (float)i,0);
	div->is_leaf = 0;
	tensor_set_require_grad(div,1);
	Tensor *mse = tensor_pairwise_mul(m,div);
	return (mse); 
}
void	optimizer_sgd(Module *module)
{
	for(int i = 0; module->parameters[i]; i++)
	{
		Tensor *lr = tensor_full(module->parameters[i]->num_dims, module->parameters[i]->shape, 0.001, 0);
		module->parameters[i] = tensor_sub(module->parameters[i], tensor_pairwise_mul(module->parameters[i]->grad, lr));
		free(lr);
	}
}
void	optimizer_step(Module *module, char *type)
{
	if (strcmp(type, "sgd") == 0)
			optimizer_sgd(module);
}
