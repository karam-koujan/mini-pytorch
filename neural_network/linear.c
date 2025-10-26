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

Tensor *Linear(int64_t in_features, int64_t out_featres, int bias, int dtype, int device)
{
    
}