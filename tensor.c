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
// (2×3) × (3×2) → (2×2)
    const int64_t shape_a[] = {2,3};
    const int64_t shape_b[] = {3,2};
    int64_t va = 1, vb = 2;
    Tensor *a = tensor_full(shape_a, 2, INT64, CPU, &va); // All 1s
    Tensor *b = tensor_full(shape_b, 2, INT64, CPU, &vb); // All 2s
    Tensor *r = tensor_matmul(a,b);
    tensor_print(r);  // Expect every element = 3 * 2 = 6
    tensor_free(a);
    tensor_free(b);
    tensor_free(r);
}