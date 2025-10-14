#include "../headers/tensor.h"
#include "../headers/print.h"




Tensor *tensor_transpose(Tensor *a, int64_t dim0, int64_t dim1)
{
    if (a == NULL)
    {
        error_msg("you entred an empty tensor");
        return (NULL);
    }
    if (dim0 < 0 || dim1 < 0)
    {
        error_msg("you entered a negative dim");
        return (NULL);
    }
    if (dim0 > a->num_dims - 1 || dim1 > a->num_dims - 1)
    {
        error_msg("you entered a dim > tensor dims");
        return (NULL);       
    }
    tensor_copy();

}