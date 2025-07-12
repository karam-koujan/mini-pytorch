#ifndef PRINT_H
#define PRINT_H
#include "./tensor.h"

void print_tensor_nbr(void *data, int index, Dtype type);
void print_tensor_recursive(void *data, int64_t *shape, int64_t *strides, int num_dims, int index, int depth, Dtype type);
void print_shape(Tensor *tensor);
void print_type(Dtype type);
void print_device(Device type);
void print_strides(Tensor *tensor);
void tensor_print(Tensor *tensor);
void    error_msg(char *msg);
#endif