#include "headers/tensor.h"
#include "headers/print.h"
#include "headers/nn.h"


int main()
{
    const int64_t shape[3] = {3,3};
    double data[3][3] = {{-1,0,5},{5,2,1},{0,0,0}};
    Tensor *a = tensor_from_arr(data, shape, 2, DOUBLE, CPU);
    tensor_set_require_grad(a, 1);
    Tensor *r = tensor_relu(a);
    tensor_backward(r, NULL);
    tensor_print(a);
    tensor_print(r);
    tensor_print(a->grad);
}


// Tensor *mse(Tensor *y, Tensor *y_pred)
// {
//     Tensor *sub = tensor_add(y, y_pred);
//     Tensor *pow = tensor_mul(sub, sub);
//     double *pow_data = pow->data;
// }