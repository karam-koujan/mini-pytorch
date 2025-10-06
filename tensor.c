#include "headers/tensor.h"
#include "headers/print.h"

// what is the difference between (7) and (1, 7)

// void    f()
// {
//     system("leaks mini_pytorch");
// }


int main()
{
    const int64_t shape_a[] = {3, 5, 2};
    Dtype type = INT32;
    Device device = CPU;
    const int64_t view[] = {-1,15};
    Tensor *a = tensor_ones(shape_a, 3, type, device);
    // tensor_print(a);
    Tensor *b = tensor_reshape(a, view, 2);
    tensor_print(b);
    tensor_infos(b);    
    // printf("tensor infos ========================\n");
    // tensor_infos(a);
    // tensor_free(a);
    tensor_free(b);
    // atexit(f);
}