#ifndef NN_H
#define NN_H

#include "../headers/tensor.h"
#include "../headers/print.h"

typedef struct Module_s
{
    Tensor **parameters;
}       Module;

size_t parameters_len(Tensor **parameters);
#endif