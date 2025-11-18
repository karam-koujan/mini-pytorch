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

Tensor *Linear(Module *m, Tensor *x, int64_t out_features, int bias, int layer)
{
    if (!m || !x)
        return (NULL);
    Dtype dtype = x->dtype == FLOAT32 ? FLOAT32 : DOUBLE; 
    int64_t in_features = x->shape[x->num_dims - 1];
    int64_t weight_shape[2] = {out_features, in_features};
    int64_t bias_shape[2] = {1,out_features};
    double k = 1 / in_features;
    Tensor *weights = parameters_len(m->parameters) <= layer ? tensor_rand(weight_shape, 2, dtype, x->device) : NULL;
    Tensor *bias_t = NULL;
    if (bias)
    {
        bias_t = biases_len(m->biases) <= layer ? tensor_rand(bias_shape, 2, dtype, x->device) :  m->biases[layer];
        if (biases_len(m->biases) <= layer)
            module_biases(m, bias_t, 1);
    }
    Tensor *weight_t = parameters_len(m->parameters) <= layer ? tensor_transpose(weights, 1, 0) : m->parameters[layer];
    if (!weight_t)
        return (printf("here\n"),tensor_free(bias_t), tensor_free(weights), NULL);
    if (parameters_len(m->parameters) <= layer)
    {
        module_parameter(m, weight_t, 1);
         tensor_free(weights);
    }
    Tensor *y = tensor_matmul(x, weight_t);
    if (!y)
        return (tensor_free(weight_t), tensor_free(weights), tensor_free(bias_t), NULL);
    if (bias)
        return tensor_add(y, bias_t);
    return y;
}