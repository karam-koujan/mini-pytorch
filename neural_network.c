#include "tensor.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#define MAX_HISTORY 2000
#define GRAPH_WIDTH 60
#define GRAPH_HEIGHT 20

#include <stdio.h>
#include <string.h>

#define MAX_HISTORY 2000
#define GRAPH_WIDTH 60
#define GRAPH_HEIGHT 20

typedef struct {
    float values[MAX_HISTORY];
    int count;
    float min;
    float max;
} CostHistory;

void cost_history_init(CostHistory *history) {
    history->count = 0;
    history->min = 1e9;
    history->max = -1e9;
}

void cost_history_add(CostHistory *history, float value) {
    if (history->count < MAX_HISTORY) {
        history->values[history->count++] = value;
        if (value < history->min) history->min = value;
        if (value > history->max) history->max = value;
    }
}

void plot_cost_ascii(CostHistory *history) {
    char graph[GRAPH_HEIGHT][GRAPH_WIDTH + 1];
    float range = history->max - history->min;
    
    // Initialize graph with spaces
    for (int i = 0; i < GRAPH_HEIGHT; i++) {
        memset(graph[i], ' ', GRAPH_WIDTH);
        graph[i][GRAPH_WIDTH] = '\0';
    }
    
    // Draw axis
    for (int i = 0; i < GRAPH_HEIGHT; i++) {
        graph[i][0] = '|';
    }
    for (int i = 0; i < GRAPH_WIDTH; i++) {
        graph[GRAPH_HEIGHT-1][i] = '-';
    }
    
    // Plot points with improved x-axis distribution
    int points_plotted[GRAPH_WIDTH] = {0};  // Track lowest y for each x position
    float min_y[GRAPH_WIDTH];  // Track minimum value for each x position
    for (int i = 0; i < GRAPH_WIDTH; i++) {
        min_y[i] = history->max;
    }
    
    // First pass: find minimum values for each x position
    for (int i = 0; i < history->count; i++) {
        int x = (int)((float)i / history->count * (GRAPH_WIDTH - 2)) + 1;
        if (x < GRAPH_WIDTH && history->values[i] < min_y[x]) {
            min_y[x] = history->values[i];
        }
    }
    
    // Second pass: plot the points
    for (int x = 1; x < GRAPH_WIDTH; x++) {
        if (min_y[x] != history->max) {
            float normalized = (min_y[x] - history->min) / range;
            int y = GRAPH_HEIGHT - 2 - (int)(normalized * (GRAPH_HEIGHT - 3));
            if (y >= 0 && y < GRAPH_HEIGHT) {
                graph[y][x] = '*';
            }
        }
    }
    
    // Connect adjacent points with lines
    for (int x = 1; x < GRAPH_WIDTH - 1; x++) {
        if (min_y[x] != history->max && min_y[x+1] != history->max) {
            float norm1 = (min_y[x] - history->min) / range;
            float norm2 = (min_y[x+1] - history->min) / range;
            int y1 = GRAPH_HEIGHT - 2 - (int)(norm1 * (GRAPH_HEIGHT - 3));
            int y2 = GRAPH_HEIGHT - 2 - (int)(norm2 * (GRAPH_HEIGHT - 3));
            
            // Draw connecting line
            int start_y = (y1 < y2) ? y1 : y2;
            int end_y = (y1 < y2) ? y2 : y1;
            for (int y = start_y + 1; y < end_y; y++) {
                if (y >= 0 && y < GRAPH_HEIGHT) {
                    graph[y][x] = '|';
                }
            }
        }
    }
    
    // Print graph with axis labels
    printf("\nCost Function Over Epochs\n");
    printf("%.4f ┐\n", history->max);
    for (int i = 0; i < GRAPH_HEIGHT; i++) {
        printf("%s\n", graph[i]);
    }
    printf("%.4f ┴", history->min);
    for (int i = 0; i < GRAPH_WIDTH-10; i++) printf("─");
    printf(" %d epochs\n", history->count);
    
    // Print epoch markers
    printf("        ");  // Align with graph
    for (int i = 0; i <= 4; i++) {
        int pos = (i * (GRAPH_WIDTH - 10) / 4);
        printf("%-12d", i * history->count / 4);
    }
    printf("\n");
    
    // Save to CSV for external plotting
    FILE *fp = fopen("cost_history.csv", "w");
    if (fp) {
        fprintf(fp, "epoch,cost\n");
        for (int i = 0; i < history->count; i++) {
            fprintf(fp, "%d,%.6f\n", i, history->values[i]);
        }
        fclose(fp);
        printf("\nCost history saved to 'cost_history.csv'\n");
    }
}


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
	int *weights_shape = (int *)malloc(a->num_dims * sizeof(int));
	if (!weights_shape)
		return NULL;
	memcpy(weights_shape,a->shape, a->num_dims);
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

Tensor *forward(Module *module,Tensor *a)
{
	Tensor *layer = Linear(module,0,a, a->shape[a->num_dims - 1], 5, 0);
	layer = Relu(layer);
	layer = Linear(module, 1,layer,layer->shape[layer->num_dims - 1],1, 0);
	return layer;
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
void	optimizer_normal(Module *module)
{
	for(int i = 0; module->parameters[i]; i++)
	{
		Tensor *lr = tensor_full(module->parameters[i]->num_dims, module->parameters[i]->shape, 0.001, 0);
		module->parameters[i] = tensor_sub(module->parameters[i], tensor_pairwise_mul(module->parameters[i]->grad, lr));
		free(lr);
	}
}
void	optimizer(Module *module, char *type)
{
	if (strcmp(type, "normal") == 0)
			optimizer_normal(module);
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
	for (int  epoch = 0; epoch < 10; epoch++)
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
	cost_history_add(&history, current_cost);  // Record the cost
	tensor_backward(prediction, cost);
	optimizer(module, "normal");
	zero_grad(module);
	free(cost);
	free(cost_tensor);
	free(prediction);
	}
	prediction = forward(module,d);	
	tensor_print(prediction);
	    plot_cost_ascii(&history);

}