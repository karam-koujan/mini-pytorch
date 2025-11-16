#include "headers/tensor.h"
#include "headers/print.h"
#include "headers/nn.h"


int main()
{
    const int64_t shape[3] = {3,2};
    Tensor *a = tensor_ones(shape, 2, FLOAT32, CPU);
    Tensor *b = tensor_ones(shape, 2, FLOAT32, CPU);
    tensor_set_require_grad(a,1);
    tensor_set_require_grad(b,1);
    Tensor *r = tensor_add(a,b);
    Tensor *l = tensor_sum(r);
    tensor_backward(l, NULL);
    tensor_print(a->grad);
    tensor_print(b->grad);
    tensor_print(r);
}


// Tensor *mse(Tensor *y, Tensor *y_pred)
// {
//     Tensor *sub = tensor_add(y, y_pred);
//     Tensor *pow = tensor_mul(sub, sub);
//     double *pow_data = pow->data;
// }