#include "../headers/nn.h"
#include <math.h>


Tensor *tensor_relu(Tensor *x)
{
    if (x->dtype != FLOAT32 && x->dtype != DOUBLE)
        return (error_msg("input tensor dtype must be a float or double"), NULL);
    Tensor *r = tensor_zeros(x->shape, x->num_dims, x->dtype, x->device);

    for(int i = 0; i < x->size; i++)
    {
        switch(x->dtype)
        {
            case FLOAT32:{
            float val = ((float *)r->data)[i];
            if (val <= 0)
                ((float *)r->data)[i] = 0.0f;
            break;
            }
            case DOUBLE :
            {
            double val = ((double *)r->data)[i];
            if (val <= 0)
                ((double *)r->data)[i] = 0.0;
            break;
            }     
        }
    }
    r->is_leaf = 0;
    return r;
}