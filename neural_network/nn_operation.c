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

void    optimizer_step(Module *module, float lr)
{
    for (int i = 0; module->parameters[i] != NULL; i++)
    {
        Tensor *param = module->parameters[i];
        Tensor *grad = param->grad;

        if (!grad) {
            continue;
        }

        for (int j = 0; j < param->size; j++)
        {
            switch(param->dtype)
            {
                case FLOAT32:
                    ((float *)param->data)[j] -= lr * ((float *)grad->data)[j];
                    break;
                case DOUBLE:
                    // Note: lr is float, cast for precision.
                    ((double *)param->data)[j] -= (double)lr * ((double *)grad->data)[j];
                    break;
                default:
                    break;
            }
        }
    }
    for (int i = 0; module->biases[i] != NULL; i++)
    {
        Tensor *param = module->biases[i];
        Tensor *grad = param->grad;

        if (!grad) {
            continue;
        }

        for (int j = 0; j < param->size; j++)
        {
            switch(param->dtype)
            {
                case FLOAT32:
                    ((float *)param->data)[j] -= lr * ((float *)grad->data)[j];
                    break;
                case DOUBLE:
                    // Note: lr is float, cast for precision.
                    ((double *)param->data)[j] -= (double)lr * ((double *)grad->data)[j];
                    break;
                default:
                    break;
            }
        }
    }
}


void    module_zero_grad(Module *module)
{
    for (int i = 0; module->parameters[i] != NULL; i++)
    {
        Tensor *grad = module->parameters[i]->grad;
        Tensor *b_grad = module->biases[i]->grad;
        module->parameters[i]->grad = tensor_zeros(grad->shape, grad->num_dims, grad->dtype, grad->device);
        module->biases[i]->grad = tensor_zeros(b_grad->shape, b_grad->num_dims, b_grad->dtype, b_grad->device);
        tensor_free(grad);
        tensor_free(b_grad);
    }
}