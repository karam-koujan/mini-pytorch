#include "headers/tensor.h"
#include "headers/print.h"

int main()
{
    const int64_t shape[] = {3, 3};
    const float arr[] = {3,2,1,5,6,8,10,13,18};
   // double val = 4;
    Tensor *t = tensor_from_arr((void *)arr, shape, 2, FLOAT32, CPU);
    tensor_print(t);
    tensor_infos(t);
}