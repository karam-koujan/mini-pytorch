#include "tensor.h"


void	f()
{
	system("leaks a.out");
}

int main()
{
	int shape[3] = {3,4,1};
	Tensor *a = tensor_empty(3,shape,0);
	free(a);
	atexit(f);
}