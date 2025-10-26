#include "../headers/nn.h"
#include <math.h>


Tensor *relu(Tensor *x)
{
    for(int i = 0; i < x->size; i++)
    {
        double val = ((double *)x->data)[i];
        if (val <= 0)
            ((double *)x->data)[i] = 0.0;
    }
    return x;
}