#ifndef NN_H
#define NN_H

#include "../headers/tensor.h"
#include "../headers/print.h"

typedef struct Module_s
{
    Tensor **parameters;
}       Module;

size_t parameters_len(Tensor **parameters);
Module *module_constructor();
Tensor *module_parameter(Module *m, Tensor *a, int requires_grad);
void    parameters_print(Tensor **parameters);
Tensor *Linear(Module *m, Tensor *x, int64_t out_features, int bias, int layer);
Tensor *tensor_relu(Tensor *x);
Tensor *calc_relu(Tensor *a);
Tensor *mse(Tensor *y, Tensor *y_pred);
void    optimizer_step(Module *module, float lr);
void    module_zero_grad(Module *module);
#endif