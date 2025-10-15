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
    const int64_t shape_a[] = {5,2,1,1,3,4};
    const int64_t shape_b[] = {1};
    Dtype type = FLOAT32;
    Device device = CPU;
    tensor_set_seed(time(NULL));
    Tensor *a = tensor_zeros(shape_a, 6, type, device);
    // tensor_print(a);
    Tensor *b = tensor_zeros(shape_b, 1, type, device);
    // tensor_print(b);
    printf("\n\n\n\n\n");
    tensor_broadcast(a,b);
    // tensor_print(a);
    tensor_print(b);
    tensor_infos(b);
    printf("\n\n\n\n\n");
    tensor_infos(a);

    // printf("tensor infos ========================\n");
    // tensor_infos(a);
    // // tensor_permute(a, view, 3);
    // tensor_free(b);
    // tensor_free(b);
    // atexit(f);
}