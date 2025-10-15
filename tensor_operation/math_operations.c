#include "../headers/tensor.h"
#include "../headers/print.h"



Tensor *tensor_add(Tensor*a, Tensor *b)
{
    if (tensor_broadcast(a,b))
        return (NULL);
   // if they have not the same datatype promote datatype
   
   
   // create a tensor_full with the newshape


   // preform the operation, I think I should create a  shared helper function that do pairwise  operations
}


Tensor *tensor_sub(Tensor*a, Tensor *b);
Tensor *tensor_matmul(Tensor*a, Tensor *b);
Tensor *tensor_mul(Tensor*a, Tensor *b);
Tensor *tensor_mm(Tensor*a, Tensor *b);
