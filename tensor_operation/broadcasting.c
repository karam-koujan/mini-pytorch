#include "../headers/tensor.h"
#include "../headers/print.h"

int is_tensor_broadcastable(Tensor *a, Tensor *b)
{
    int i = a->num_dims - 1;
    int j = b->num_dims - 1;
    int k = i > j ? j : i;
    while (k >= 0)
    {
        if (a->shape[i] != b->shape[j] && b->shape[j] != 1 && a->shape[i] != 1)
            return (error_msg("the tensors are not broadcastable") ,0);
        i--;
        j--;
        k--;
    }
    return (1);
}
