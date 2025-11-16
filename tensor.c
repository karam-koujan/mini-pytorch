#include "headers/tensor.h"
#include "headers/print.h"
#include "headers/nn.h"


int main()
{
    const int64_t shape[3] = {3,2};
    Tensor *a = tensor_ones(shape, 2, INT64, CPU);
    tensor_print(a);
    Tensor *r = tensor_mean(a);
    tensor_print(r);
}


// Tensor *mse(Tensor *y, Tensor *y_pred)
// {
//     Tensor *sub = tensor_add(y, y_pred);
//     Tensor *pow = tensor_mul(sub, sub);
//     double *pow_data = pow->data;
// }