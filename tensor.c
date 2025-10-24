#include "headers/tensor.h"
#include "headers/print.h"

// what is the difference between (7) and (1, 7)

// void    f()
// {
//     system("leaks mini_pytorch");
// }
#include "time.h"

int main()
{
    const int64_t shape_a[] = {1,1,2};
    const int64_t shape_b[] = {2,2,1};
    double va = 2.0, vb = 5.0;
    Tensor *a = tensor_full(shape_a, 3, DOUBLE, CPU, &va);
    Tensor *b = tensor_full(shape_b, 3, DOUBLE, CPU, &vb);
    tensor_set_require_grad(a, 1);
    tensor_set_require_grad(b,1);
    Tensor *r = tensor_div(a,b);
    // tensor_set_require_grad(r, 1);
    // Tensor *p = tensor_mul(r, r);
    // tensor_backward(p, NULL);
    tensor_print(r);
    printf("here is grad\n");
    tensor_print(a->grad);
    tensor_print(b->grad);
    tensor_free(a);
    tensor_free(b);
    tensor_free(r);
    // tensor_print(a);
}