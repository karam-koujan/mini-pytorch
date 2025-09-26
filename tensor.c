#include "headers/tensor.h"
#include "headers/print.h"

// what is the difference between (7) and (1, 7)

void    f()
{
    system("leaks mini_pytorch");
}


int main()
{
    const int64_t shape_a[] = {4,1,6,2};
    Dtype type = INT32;
    Device device = CPU;
    const int64_t view[] = {-1};
    Tensor *a = tensor_ones(shape_a, 4, type, device);
    tensor_print(a);
    tensor_view(a, view, 1);
    // printf("tensor infos ========================\n");
    // tensor_infos(a);
    // tensor_infos(b);
    // tensor_free(a);
    // tensor_free(b);
    // atexit(f);
}