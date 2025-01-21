#include "tensor.h"
#include "plot_loss.h"


Tensor *forward(Module *module,Tensor *a)
{
	Tensor *layer = Linear(module,0,a, a->shape[a->num_dims - 1], 5, 0);
	layer = Relu(layer);
	layer = Linear(module, 1,layer,layer->shape[layer->num_dims - 1],1, 0);
	return layer;
}

int main()
{
	tensor_set_seed(1337);
	CostHistory history;
	cost_history_init(&history);
	float data[20][2] = {
	    {0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}, {0.7, 0.8}, {0.2, 0.3},
	    {0.4, 0.6}, {0.6, 0.8}, {0.8, 0.9}, {0.1, 0.4}, {0.3, 0.5},
	    {0.5, 0.7}, {0.7, 0.9}, {0.2, 0.5}, {0.4, 0.7}, {0.6, 0.9},
	    {0.1, 0.3}, {0.3, 0.6}, {0.5, 0.8}, {0.7, 0.7}, {0.9, 0.8}
	};

	// Labels follow the pattern (x₁ + 2*x₂)/3 with small variations
	float labels[20] = {
	    0.17, 0.37, 0.57, 0.77, 0.27,
	    0.53, 0.73, 0.87, 0.30, 0.43,
	    0.63, 0.83, 0.40, 0.60, 0.80,
	    0.23, 0.50, 0.70, 0.70, 0.83
	};
	int data_shape[] = {1,20,2};
	Tensor *d = tensor_tensor(data,data_shape,3);
	int label_shape[] = {20,1};
	Tensor *l = tensor_tensor(labels,label_shape,3);
	Module *module = nn();	
	Tensor *prediction;
	for (int  epoch = 0; epoch < 2000; epoch++)
	{
	printf("=========== epoch : %i =================\n",epoch);
	prediction = forward(module, d);
	if (epoch == 0)
	{
		printf("before training prediction\n");
		tensor_print(prediction);
	}
	Tensor *cost_tensor = mse(prediction, l);
	Tensor *cost = tensor_full(prediction->num_dims,prediction->shape,((float *)cost_tensor->data)[0],0);
    float current_cost = ((float *)cost_tensor->data)[0];
	if (epoch % 10 == 0)
		cost_history_add(&history, current_cost);  // Record the cost
	tensor_backward(prediction, cost);
	optimizer_step(module, "sgd");
	zero_grad(module);
	free(cost);
	free(cost_tensor);
	free(prediction);
	}
	prediction = forward(module,d);	
	tensor_print(prediction);
	plot_cost_ascii(&history);

}