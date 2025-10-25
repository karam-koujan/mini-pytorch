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
    const int64_t shape_b[] = {1,1,2};
    double va = 2.0, vb = 5.0;
    Tensor *a = tensor_full(shape_a, 3, DOUBLE, CPU, &va);
    Tensor *b = tensor_full(shape_b, 3, DOUBLE, CPU, &vb);
    
    tensor_free(a);
    tensor_free(b);
 
}