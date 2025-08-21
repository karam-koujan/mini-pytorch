#include "headers/tensor.h"
#include "headers/print.h"

// what is the difference between (7) and (1, 7)
int main()
{
    const int64_t shape_a[] = {3, 0, 4};
    const int64_t shape_b[] = {1, 0, 4};
    Dtype type = INT32;
    Device device = CPU;
    Tensor *a = tensor_ones(shape_a, 3, type, device);
    Tensor *b = tensor_ones(shape_b, 3, type, device);
    tensor_print(a);
    tensor_print(b);
    printf("is broadcastable %i", is_tensor_broadcastable(a, b));
}