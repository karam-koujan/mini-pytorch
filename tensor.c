#include "headers/tensor.h"
#include "headers/print.h"

// what is the difference between (7) and (1, 7)
int main()
{
   const int64_t shape_a[] = {3, 1};
    const int64_t shape_b[] = {3, 4};
    Dtype type = INT32;
    Device device = CPU;
    Tensor *a = tensor_ones(shape_a, 2, type, device);
    // Tensor *b = tensor_ones(shape_b, 2, type, device);
    // int64_t d = 5;
    // Tensor *a = tensor_scalar(&d, type, device);
    Tensor *b = tensor_ones(shape_b, 2, type, device); 
    tensor_print(a);
    tensor_print(b);
    printf("tensor_broadcast return val %i\n", tensor_broadcast(a, b));
    tensor_print(a);
    tensor_print(b);
    printf("tensor infos ========================\n");
    tensor_infos(a);
    tensor_infos(b);
}