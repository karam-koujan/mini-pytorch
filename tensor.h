#ifndef TENSOR_H
#define TENSOR_H

#include <stdarg.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


enum Dtype
{
	FLOAT32,
	DOUBLE,
	INT32,
	INT64
};

enum Device
{
	CPU,
	GPU
};


typedef	struct
{
	void *data;
	int *shape;
	int *strides;
	enum Dtype dtype;
	enum Device device;
	void *grad;
	int	requires_grad;
	int num_dims;
	int	is_leaf;
	void *grad_fn;
} Tensor;

typedef struct
{
	Tensor **saved_tensors;
	void *(**next_functions)(Tensor*,Tensor*);
}	Grad_Node;
Tensor *tensor_rand(int dim,...);
void tensor_set_seed(unsigned int seed);
float	generate_random();
void tensor_print(Tensor *tensor);
Tensor *tensor_full(int dim,...);
Tensor *tensor_ones(int dim,...);
Tensor *tensor_zeros(int dim,...);
Tensor	 *tensor_empty(int dim,...);
void	tensor_fill(Tensor *tensor, float num);
int		tensor_entries_len(Tensor *tensor);
float	*create_empty_data(int dim,int *shape);
void	add_options(va_list arg,Tensor *tensor);
int	*create_stride(int num_dims, int *shape);
int	*create_shape(va_list arg,int dim);
int tensor_validate_shape(Tensor *a, Tensor *b);
int	tensor_is_broadcastable(Tensor *a,Tensor *b, char type);
float	tensor_get_num(Tensor *a,...);
Tensor *tensor_reshape(Tensor *a,int num_dim, int *shape);
Tensor *tensor_pairwise_mul(Tensor *a, Tensor *b);
Tensor *tensor_matmul(Tensor *a, Tensor *b);
Tensor *tensor_div(Tensor *a, Tensor *b);
Tensor *tensor_sub(Tensor *a, Tensor *b);
Tensor *tensor_add(Tensor *a, Tensor *b);
ssize_t	tensor_size(Tensor *a, ssize_t num_dims);
float	*tensor_contigous(Tensor *a, int *new_shape);
Tensor *tensor_t(Tensor *a);
Tensor *tensor_transpose(Tensor *a, int dim0, int dim1);
void *tensor_backmatmul(Tensor *a, Tensor *b);

#endif