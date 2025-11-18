#include "headers/tensor.h"
#include "headers/print.h"
#include "headers/nn.h"



Tensor *forward(Module *module, Tensor *x)
{
    Tensor *l1 = Linear(module, x, 2, 0);
    Tensor *l2 = Linear(module, l1, 2, 0);
    Tensor *ypred = Linear(module, l1, 1, 0);
    return ypred;
}

int main()
{
    float data[20][2] = {
        {0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}, {0.7, 0.8}, {0.2, 0.3},
        {0.4, 0.6}, {0.6, 0.8}, {0.8, 0.9}, {0.1, 0.4}, {0.3, 0.5},
        {0.5, 0.7}, {0.7, 0.9}, {0.2, 0.5}, {0.4, 0.7}, {0.6, 0.9},
        {0.1, 0.3}, {0.3, 0.6}, {0.5, 0.8}, {0.7, 0.7}, {0.9, 0.8}
    };  
    float labels[20] = {
        0.17, 0.37, 0.57, 0.77, 0.27,
        0.53, 0.73, 0.87, 0.30, 0.43,
        0.63, 0.83, 0.40, 0.60, 0.80,
        0.23, 0.50, 0.70, 0.70, 0.83
    };
    int data_shape[] = {1,20,2};
    int label_shape[] = {1,20};
    Dtype dtype = FLOAT32;
    Device device = CPU;
    Tensor *data_t = tensor_from_arr(data, data_shape, 3, dtype, device);
    Tensor *label_t = tensor_from_arr(labels, label_shape,2, dtype, device);
    Module *module = module_constructor();
    Tensor *pred = forward(module, data_t);
    tensor_backward(pred, NULL);
}


