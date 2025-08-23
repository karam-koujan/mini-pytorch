#include "headers/tensor.h"
#include "headers/print.h"

// what is the difference between (7) and (1, 7)

void    f()
{
    system("leaks mini_pytorch");
}


int main()
{
    const int64_t shape_a[] = {2};
    const int64_t shape_b[] = {3, 2};
    Dtype type = INT32;
    Device device = CPU;
    Tensor *a = tensor_ones(shape_a, 1, type, device);
    Tensor *b = tensor_ones(shape_b, 2, type, device);
    tensor_print(a);
    tensor_print(b);
    printf("tensor_broadcast return val %i\n", tensor_broadcast(a, b));
    tensor_print(a);
    tensor_print(b);
    // printf("tensor infos ========================\n");
    // tensor_infos(a);
    // tensor_infos(b);
    // tensor_free(a);
    // tensor_free(b);
    // atexit(f);
    
}