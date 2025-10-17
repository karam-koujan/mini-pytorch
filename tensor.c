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
    const int64_t shape_a[] = {1,3,2};
    // Dtype type = FLOAT32;
    Device device = CPU;
    // tensor_set_seed(time(NULL));
    double val_a = 10;
    Tensor *a = tensor_full(shape_a, 3, DOUBLE, device, &val_a);
    Tensor *b = tensor_deep_copy(a);
    tensor_print(a);
    tensor_print(a);
    tensor_free(a);
    tensor_print(b);
}