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
// (3) × (3×2) → (2)
    const int64_t shape_a[] = {1,1,2};
    const int64_t shape_b[] = {1,2,3};
    int64_t va = 1, vb = 5;
    Tensor *a = tensor_full(shape_a, 3, INT64, CPU, &va);
    Tensor *b = tensor_full(shape_b, 3, INT64, CPU, &vb);
    tensor_set_require_grad(a, 1);
    tensor_set_require_grad(b,1);
    Tensor *r = tensor_matmul(a,b);
    tensor_backward(r, NULL);
    tensor_print(r);
    printf("here is grad\n");
    tensor_print(a->grad);
    tensor_print(b->grad);
    tensor_free(a);
    tensor_free(b);
    tensor_free(r);
// EXPECT each final 3x3 matrix filled with 3*2 + 3*2 (sum over 2 dims) = 12
}

/*

Tensor of shape (1,1,2):
[[1,1]
]


Tensor of shape (1,2,3):
[[5,5,5]
,[5,5,5]
]


*/


/*

// (3) × (3×2) → (2)
const int64_t shape_a[] = {3};
const int64_t shape_b[] = {3,2};
int64_t va = 1, vb = 5;
Tensor *a = tensor_full(shape_a, 1, INT64, CPU, &va);
Tensor *b = tensor_full(shape_b, 2, INT64, CPU, &vb);
Tensor *r = tensor_matmul(a,b);
tensor_print(r);  // expect [15,15]

*/