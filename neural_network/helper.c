#include "../headers/nn.h"




size_t parameters_len(Tensor **parameters)
{
    size_t count = 0;
    if (!parameters)
        return (count);
    for (size_t i = 0; parameters[i] != NULL; i++)
    {
        count++;
    }
    return (count);
}






