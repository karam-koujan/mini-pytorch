#include "tensor.h"


void	f()
{
	system("leaks a.out");
}

int main()
{
	int shape[4] = {3,4,1,5};
	Tensor *a = tensor_full(4,shape,0.8,0);
	tensor_print(a);
	tensor_clean(a);
	atexit(f);
}