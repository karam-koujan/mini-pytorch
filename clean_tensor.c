#include "tensor.h"




void	tensor_clean(Tensor *a)
{
	free(a->data);
	free(a->shape);
	free(a->strides);
	free(a->grad);
	free(a->grad_fn);
	free(a);
}