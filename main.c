#include "headers/tensor.h"
#include "headers/print.h"
#include "headers/nn.h"
#include <time.h>

void    print_logs(Tensor *pred, Tensor *label_t, Tensor *loss, int epoch, int i)
{
    if (i % 500 == 0)
    {
        printf("epoch: %i\n", epoch);
        printf("loss func\n");
        tensor_print(loss);
    }
    if (i == epoch || i == 0)
    {
        printf("epoch: %i\n", epoch);
        printf("true label \n");
        tensor_print(label_t);
        printf("prediced label \n");
        tensor_print(pred);
        printf("loss func\n");
        tensor_print(loss);
    }
}

int main()
{
    tensor_set_seed(1337);
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
    int64_t data_shape[] = {1,20,2};
    int64_t label_shape[] = {20,1};
    Dtype dtype = FLOAT32;
    Device device = CPU;
    Tensor *data_t = tensor_from_arr(data, data_shape, 3, dtype, device);
    Tensor *label_t = tensor_from_arr(labels, label_shape,2, dtype, device);
    Module *module = module_constructor();
    int epoch = 2;
    float lr = 0.001;
    for (int i = 0; i <= epoch; i++)
    {
        // forward operations

        Tensor *l1 = Linear(module, data_t, 2, 1, 0);
        Tensor *r = tensor_relu(l1);
        Tensor *l2 = Linear(module, r, 2, 1, 1);
        Tensor *r2 = tensor_relu(l2);
        Tensor *pred = Linear(module, r2, 1, 1, 2);        

        // Loss function

        Tensor *sub = tensor_sub(label_t, pred);
        Tensor *pow = tensor_mul(sub, sub);
        Tensor *loss = tensor_sum(pow);
        print_logs(pred, label_t, loss, epoch, i);

        // back propagation

        tensor_backward(loss, NULL);

        // weight update

        optimizer_step(module, lr);

        module_zero_grad(module);
        tensor_free(l1);
        tensor_free(r);      // <-- Leaked tensor from first ReLU
        tensor_free(l2);
        tensor_free(r2);     // <-- Leaked tensor from second ReLU
        tensor_free(sub);    // <-- This was also leaking
        tensor_free(pow);    // <-- This was also leaking
        tensor_free(pred);
        tensor_free(loss);
    }
    tensor_free(label_t);
    tensor_free(data_t);
    module_free(module);
}


