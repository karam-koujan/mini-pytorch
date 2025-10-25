#include "../headers/nn.h"





Tensor *Parameter(Module *m, Tensor *a)
{
    size_t size;
    if (!m->parameters)
    {
        size = 2 * sizeof(Tensor *);
    }else
    {
        size = (parameters_len(m->parameters) + 1) * sizeof(Tensor *);
    }

}