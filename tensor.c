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
// (2×2×3) × (2×3×2) → (2×2×2)
    const int64_t shape_a[] = {2,2,3};
    const int64_t shape_b[] = {2,3,2};
    int64_t va = 1, vb = 2;
    Tensor *a = tensor_full(shape_a, 3, INT64, CPU, &va);
    Tensor *b = tensor_full(shape_b, 3, INT64, CPU, &vb);
    Tensor *r = tensor_matmul(a,b);
    tensor_print(r);  // Expect each 2x2 slice filled with 3*2 = 6
    tensor_free(a);
    tensor_free(b);
    tensor_free(r);
}