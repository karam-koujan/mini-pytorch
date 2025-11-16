#include "headers/tensor.h"
#include "headers/print.h"
#include "headers/nn.h"


int main()
{
    const int64_t shape[3] = {3,2};
    float d = 5.0;
    Tensor *a = tensor_full(shape, 2, FLOAT32, CPU, &d);
    Tensor *b = tensor_full(shape, 2, FLOAT32, CPU, &d);
    tensor_set_require_grad(a,1);
    tensor_set_require_grad(b,1);
    Tensor *r = tensor_mul(a,b);
    Tensor *l = tensor_mean(r);
    tensor_backward(l, NULL);
    tensor_print(l);
    tensor_print(a->grad);
    tensor_print(b->grad);
}


// Tensor *mse(Tensor *y, Tensor *y_pred)
// {
//     Tensor *sub = tensor_add(y, y_pred);
//     Tensor *pow = tensor_mul(sub, sub);
//     double *pow_data = pow->data;
// }