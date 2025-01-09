#include "tensor.h"
#include <math.h>

Tensor *Linear(Tensor *a, int in_features, int out_features, int use_bias)
{
	int *weights_shape = (int *)malloc(a->num_dims * sizeof(int));
	if (!weights_shape)
		return NULL;
	memcpy(weights_shape,a->shape, a->num_dims);
	weights_shape[a->num_dims - 1] = in_features;
	weights_shape[a->num_dims - 2] = out_features;
	Tensor *weights = tensor_ones(a->num_dims, weights_shape,0);
	tensor_set_require_grad(weights,1);
	Tensor *weights_t = tensor_t(weights);
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

Tensor *foward(Tensor *a)
{
	Tensor *layer = Linear(a, a->shape[a->num_dims - 1], 5, 0);
	layer = Relu(layer);
	layer = Linear(layer, layer->shape[layer->num_dims - 1], 5, 0);
	layer = Relu(layer);
	layer = Linear(layer, layer->shape[layer->num_dims - 1], 5, 0);
	layer = Relu(layer);
	return Linear(layer, layer->shape[layer->num_dims - 1], 1, 0);
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
	tensor_print(d);
	tensor_print(l);
	Tensor *prediction;
	for(int epoch = 0; epoch < 50; epoch++)
	{
		prediction = foward(d);
	}
	tensor_print(prediction);
	Tensor *cost_fn = cost(prediction, l);
	tensor_backward(cost_fn, NULL);
	tensor_print(cost_fn);
}