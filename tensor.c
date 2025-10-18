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
    const int64_t shape_a[] = {2,2,3,2};
    const int64_t shape_b[] = {1,1,2,3};

    Dtype type = INT64;
    Device device = CPU;
    // tensor_set_seed(time(NULL));
    int64_t val_a = 3;
    int64_t val_b = 2;
    Tensor *a = tensor_full(shape_a, 4, type, device, &val_a);
    Tensor *b = tensor_full(shape_b, 4, type, device, &val_b);
    Tensor *result = tensor_matmul(a, b);
    tensor_print(result);
    tensor_infos(result);
}