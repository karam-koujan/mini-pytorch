#include "../headers/nn.h"
#include <math.h>




Tensor *calc_relu(Tensor *a)
{
    Tensor *r = tensor_deep_copy(a);
    for(int i = 0; i < a->size; i++)
    {
        switch(a->dtype)
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
            default :
                return (error_msg("input tensor dtype must be a float or double"), tensor_free(r),NULL);

        }
    }
    return r;
}

Tensor *tensor_relu(Tensor *x)
{
    if (x->dtype != FLOAT32 && x->dtype != DOUBLE)
        return (error_msg("input tensor dtype must be a float or double"), NULL);
    Tensor *r = calc_relu(x);
    r->grad_fn = x->requires_grad ? create_relu_node(x) : NULL;
    r->is_leaf = 0;
    if (x->requires_grad)
        tensor_set_require_grad(r, 1);
    return r;
}

Tensor *mse(Tensor *y, Tensor *y_pred)
{
    Tensor *sub = tensor_sub(y, y_pred);
    Tensor *pow = tensor_mul(sub, sub);
    Tensor *sum = tensor_sum(pow);
    return sum;
}