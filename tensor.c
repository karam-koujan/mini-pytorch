#include "headers/tensor.h"
#include "headers/print.h"
#include "headers/nn.h"

// what is the difference between (7) and (1, 7)

// void    f()
// {
//     system("leaks mini_pytorch");
// }

int main()
{
    const int64_t shape_a[] = {1,1,2};
    const int64_t shape_b[] = {1,1,2};
    double va = 2.0, vb = 5.0;
    Tensor *a = tensor_full(shape_a, 3, DOUBLE, CPU, &va);
    Tensor *b = tensor_full(shape_b, 3, DOUBLE, CPU, &vb);
    Module *m  = module_constructor();
    module_parameter(m, a, 1);
    module_parameter(m, b, 1);
        module_parameter(m, b, 1);
        module_parameter(m, b, 1);
        module_parameter(m, b, 1);
        module_parameter(m, b, 1);
        module_parameter(m, b, 1);
        module_parameter(m, b, 1);
        module_parameter(m, b, 1);



    parameters_print(m->parameters);
    tensor_free(a);
    tensor_free(b);
    free(m->parameters);
    free(m);
}