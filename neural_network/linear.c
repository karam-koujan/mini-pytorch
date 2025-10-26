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

Tensor *Linear(Module *m, Tensor *x, int64_t out_features, int bias, int dtype, int device)
{
    if (!m || !x)
        return (NULL);
    int dtype = x->dtype == FLOAT32 ? FLOAT32 : DOUBLE; 
    int64_t in_features = x->shape[x->num_dims - 1];
    int64_t weight_shape[2] = {in_features , out_features};
    int64_t bias_shape[2] = {out_features, 1};
    double k = 1 / in_features;
    Tensor *weights = tensor_urand(weight_shape, 2, dtype, x->device, -sqrt(k), sqrt(k));
    if (!weights)
        return (NULL);
    Tensor *bias_t = NULL;
    if (bias)
    {
        bias_t = tensor_urand(bias_shape, 2, dtype, x->device,-sqrt(k), sqrt(k));
        if (!bias_t)
            return (tensor_free(weights), NULL);
        bias_t = module_parameter(m, bias_t, 1);
    }
    Tensor *weight_t = tensor_transpose(weight_t, 1, 0);
    if (!weight_t)
        return (tensor_free(bias), tensor_free(weights), NULL);
    module_parameter(m, weight_t, 1);
    Tensor *y = tensor_matmul(x, weight_t);
    if (!y)
        return (tensor_free(weight_t), tensor_free(weights), tensor_free(bias_t));
    tensor_free(weights);
    if (bias)
        return tensor_add(y, bias_t);
    return y;
}