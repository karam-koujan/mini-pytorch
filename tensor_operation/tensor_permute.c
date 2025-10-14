#include "../headers/tensor.h"
#include "../headers/print.h"

Tensor *tensor_permute(Tensor *a, int64_t *dims, int64_t num_dims)
{
    if (a == NULL)
    {
        error_msg("you entred an empty tensor");
        return (NULL);
    }
    if (num_dims != a->num_dims)
    {
        error_msg("the ordering dimensions should have the same numbers as the original tensor");
        return (NULL);
    }
    for (int64_t i = 0 ; i < num_dims; i++)
    {
        if (dims[i] < 0)
        {
            error_msg("you entered a negative dim");
            return (NULL);            
        }
        if (dims[i] > num_dims - 1)
        {
            error_msg("you entered a dim > tensor dims");
            return (NULL);  
        }

    }
}