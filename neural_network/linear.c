#include "../headers/nn.h"





Tensor *module_parameter(Module *m, Tensor *a, int requires_grad)
{
    size_t nmemb = parameters_len(m->parameters);
    if (!a->requires_grad && requires_grad)
    {
        tensor_set_require_grad(a, 1);
    }
    Tensor **new_parameters;
    if (nmemb == 0)
    {
        nmemb = 1;
    }else
    {
        nmemb = nmemb + 1;
    }
    new_parameters = calloc(nmemb + 1, sizeof(Tensor *));
    if (!new_parameters)
        return (NULL);
    if (!m->parameters)
    {
        new_parameters[0] = a;
        new_parameters[1] = NULL;
    }
    else
    {
        memcpy(new_parameters, m->parameters, (nmemb) * sizeof(Tensor *));
        new_parameters[nmemb - 1] = a;
        new_parameters[nmemb ]  = NULL;
    }
    free(m->parameters);
    m->parameters = new_parameters;
    return a;
}

Tensor *module_biases(Module *m, Tensor *a, int requires_grad)
{
    size_t nmemb = biases_len(m->biases);
    if (!a->requires_grad && requires_grad)
    {
        tensor_set_require_grad(a, 1);
    }
    Tensor **new_biases;
    if (nmemb == 0)
    {
        nmemb = 1;
    }else
    {
        nmemb = nmemb + 1;
    }
    new_biases = calloc(nmemb + 1, sizeof(Tensor *));
    if (!new_biases)
        return (NULL);
    if (!m->biases)
    {
        new_biases[0] = a;
        new_biases[1] = NULL;
    }
    else
    {
        memcpy(new_biases, m->biases, (nmemb) * sizeof(Tensor *));
        new_biases[nmemb - 1] = a;
        new_biases[nmemb ]  = NULL;
    }
    free(m->biases);
    m->biases = new_biases;
    return a;
}

void    parameters_print(Tensor **parameters)
{
    if (!parameters)
        return ;
    size_t i = -1;
    while (parameters[++i])
        tensor_print(parameters[i]);
}

Module *module_constructor()
{
    Module *m = malloc(sizeof(Module));
    if (!m)
        return (NULL);
    m->parameters = NULL;
    return (m);
}

// Tensor *get_weight(Module *m, Tensor *x, int64_t out_features, int bias, int layer)
// {
//     Dtype dtype = x->dtype == FLOAT32 ? FLOAT32 : DOUBLE; 
//     int64_t in_features = x->shape[x->num_dims - 1];
//     int64_t weight_shape[2] = {out_features, in_features};
//     int par_idx = bias ? 1 : 0; 
//     Tensor *weights = parameters_len(m->parameters) < (layer + par_idx) * 2 ? tensor_rand(weight_shape, 2, dtype, x->device) : NULL;

// }

/*




*/

Tensor *Linear(Module *m, Tensor *x, int64_t out_features, int bias, int layer)
{
    if (!m || !x)
        return (NULL);
    Dtype dtype = x->dtype == FLOAT32 ? FLOAT32 : DOUBLE; 
    int64_t in_features = x->shape[x->num_dims - 1];
    int64_t weight_shape[2] = {in_features, out_features};
    int64_t bias_shape[2] = {1,out_features};
    int step = bias ? 2 : 1;
    int weight_idx = layer * step;
    int bias_idx   = layer * step + 1;
    double xavier = 6.0 / (in_features + out_features);
    double limit = sqrt(xavier);
    Tensor *weights = parameters_len(m->parameters) <= weight_idx
                        ? tensor_urand(weight_shape, 2, dtype, x->device, -limit, limit)
                        : m->parameters[weight_idx];


    if (parameters_len(m->parameters) <= weight_idx)
        module_parameter(m, weights, 1);

    // Bias
    Tensor *bias_t = NULL;
    if (bias)
    {
        if (parameters_len(m->parameters) <= bias_idx)
            bias_t = tensor_urand(bias_shape, 2, dtype, x->device, -limit, limit);
        else
            bias_t = m->parameters[bias_idx];

        if (parameters_len(m->parameters) <= bias_idx)
            module_parameter(m, bias_t, 1);
    }
    Tensor *y = tensor_matmul(x, weights);
    if (!y)
        return (tensor_free(weights), tensor_free(bias_t), NULL);
    if (bias)
        return tensor_add(y, bias_t);
    return y;
}