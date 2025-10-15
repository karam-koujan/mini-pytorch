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
    const int64_t shape_a[] = {3,2,1,1,5};
    const int64_t shape_b[] = {3,1,1,1,5};
    Dtype type = FLOAT32;
    Device device = CPU;
    tensor_set_seed(time(NULL));
    Tensor *a = tensor_zeros(shape_a, 5, type, device);
    Tensor *b = tensor_zeros(shape_b, 5, type, device);
    printf("\n\n\n\n\n");
    tensor_broadcast(a,b);
    tensor_infos(a);
    printf("\n\n\n\n\n");
    tensor_infos(b);
    tensor_free(a);
    tensor_free(b);
}