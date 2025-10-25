#include "../headers/nn.h"





Tensor *Parameter(Module *m, Tensor *a, int requires_grad)
{
    size_t nmemb = parameters_len(m->parameters);
    if (!a->requires_grad && requires_grad)
    {
        tensor_set_require_grad(a, 1);
    }
    Tensor **new_parameters;
    if (nmemb == 0)
    {
        nmemb = 2;
    }else
    {
        nmemb = nmemb + 1;
    }
    new_parameters = calloc(nmemb, sizeof(Tensor *));
    if (!new_parameters)
        return (NULL);
    if (!m->parameters)
    {
        new_parameters[0] = a;
        new_parameters[1] = NULL;
    }
    else
    {
        memcpy(new_parameters, m->parameters, (nmemb - 1) * sizeof(Tensor *));
        new_parameters[nmemb - 2] = a;
        new_parameters[nmemb - 1]  = NULL;
    }
    free(m->parameters);
    m->parameters = new_parameters;
    return a;
}

Module *module_constructor()
{
    Module *m = malloc(sizeof(Module));
    if (!m)
        return (NULL);
    m->parameters = NULL;
    return (m);
}