#include "headers/tensor.h"
#include "headers/print.h"
#include "headers/nn.h"


int main()
{
    const int64_t shape[3] = {1,3,3};
    const int64_t shape_b[2] = {3,2};
    float d = 5.0;
    float c = 1.0;
    Tensor *a = tensor_full(shape, 3, FLOAT32, CPU, &d);
    Tensor *b = tensor_full(shape_b, 2, FLOAT32, CPU, &c);
    tensor_set_require_grad(a,1);
    tensor_set_require_grad(b,1);
    Tensor *l = tensor_matmul(a,b);
    Tensor *r = tensor_relu(l);
    Tensor *j = tensor_sum(r);
    tensor_backward(j, NULL);
    tensor_print(j);
    tensor_print(a->grad);
    tensor_print(b->grad);
    tensor_free(a);
    tensor_free(b);
    tensor_free(l);
    tensor_free(j);
}


