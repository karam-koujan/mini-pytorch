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
    const int64_t shape_a[] = {3,2,5};
    const int64_t shape_b[] = {2,5};
    Dtype type = FLOAT32;
    Device device = CPU;
    tensor_set_seed(time(NULL));
    Tensor *a = tensor_zeros(shape_a, 3, type, device);
    tensor_print(a);
    Tensor *b = tensor_zeros(shape_b, 2, type, device)
    // tensor_print(b);
    // tensor_infos(b);    
    // printf("tensor infos ========================\n");
    // tensor_infos(a);
    // tensor_permute(a, view, 3);
    tensor_free(b);
    // tensor_free(b);
    // atexit(f);
}