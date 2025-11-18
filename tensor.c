#include "headers/tensor.h"
#include "headers/print.h"
#include "headers/nn.h"


int main()
{
    const int64_t shape[3] = {1,3,3};
    const int64_t shape_b[2] = {3,3};
    float d = 5.0;
    float c = 1.0;
    Tensor *a = tensor_full(shape, 3, FLOAT32, CPU, &d);
    Tensor *b = tensor_full(shape_b, 2, FLOAT32, CPU, &c);
    tensor_set_require_grad(a,1);
    tensor_set_require_grad(b,1);
    Tensor *l = tensor_mul(a,b);
    Tensor *z = tensor_add(l, b);

    Tensor *j = tensor_sum(z);
    tensor_backward(j, NULL);
    tensor_print(j);
    tensor_print(a->grad);
    tensor_print(b->grad);
    tensor_free(a);
    tensor_free(b);
    tensor_free(l);
    tensor_free(j);
}


// Tensor *mse(Tensor *y, Tensor *y_pred)
// {
//     Tensor *sub = tensor_add(y, y_pred);
//     Tensor *pow = tensor_mul(sub, sub);
//     double *pow_data = pow->data;
// }