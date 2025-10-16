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
    const int64_t shape_b[] = {1,3,2};
    // Dtype type = FLOAT32;
    Device device = CPU;
    // tensor_set_seed(time(NULL));
    float val_a = 10;
    int val_b = 2;
    Tensor *a = tensor_full(shape_a, 3, FLOAT32, device, &val_a);
    Tensor *b = tensor_full(shape_b, 3, INT32, device, &val_b);
    tensor_print(a);
    tensor_print(b);
    printf("\n\n\n\n\n");
    Tensor *r = tensor_div(a,b);
    tensor_print(r);
    tensor_infos(r);
}