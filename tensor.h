#ifndef TENSOR_H
#define TENSOR_H

#include <stdarg.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

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
	int size;
	int	requires_grad;
	int num_dims;
	int	is_leaf;
	void *grad_fn;
} Tensor;

typedef struct Node
{
	Tensor *grad;
	Tensor **saved_tensors;
	Tensor **(*calculate_gradient)(struct Node *node,Tensor *grad);
}	Grad_Node;

typedef struct
{
	Tensor **parameters;
} Module;

Tensor *tensor_rand(int dim,int *shape,...);
void tensor_set_seed(unsigned int seed);
float	generate_random();
void tensor_print(Tensor *tensor);
Tensor *tensor_full(int dim,int *shape,...);
Tensor *tensor_ones(int dim,int *shape,...);
Tensor *tensor_zeros(int dim,int *shape,...);
Tensor	 *tensor_empty(int dim,int *shape,...);
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
Tensor **tensor_backmatmul(Grad_Node *node, Tensor *grad);
Tensor	**tensor_broadcast(Tensor *a, Tensor *b, char type);
int tensor_is_contigious(Tensor *a);
Tensor *tensor_detach(Tensor *a);
Grad_Node	*create_matmul_node(Tensor *a, Tensor *b);
void	tensor_accumulate_grad(Tensor *a, Tensor *grad);
void	tensor_set_require_grad(Tensor *a, int require_grad);
Grad_Node	*create_add_node(Tensor *a, Tensor *b);
Tensor **tensor_backadd(Grad_Node *node, Tensor *grad);
Tensor *tensor_mm(Tensor *a,Tensor *b);
Grad_Node	*create_mm_node(Tensor *a, Tensor *b);
Tensor **tensor_backmm(Grad_Node *node, Tensor *grad);
Tensor *tensor_collapse(Tensor *a, int *original_shape,int new_dim);
Tensor *tensor_tensor(void *data, int *shape, int dims);
void	tensor_backward(Tensor *a, Tensor *prev_grad);
Grad_Node	*create_sub_node(Tensor *a, Tensor *b);
Tensor **tensor_backsub(Grad_Node *node, Tensor *grad);
void	module_param_add(Module *module,Tensor *a);
Tensor **tensor_backpairwise_mul(Grad_Node *node, Tensor *grad);
Grad_Node	*create_pairwise_mul_node(Tensor *a, Tensor *b);
void	tensor_clean(Tensor *a);
Tensor *tensor_init(int dim, int *shape);
void	optimizer(Module *module, char *type);
Tensor *mse(Tensor *pred, Tensor *label);
void	zero_grad(Module *module);
void	module_param_add(Module *module,Tensor *a);
Tensor *Relu(Tensor *a);
Module *nn();
Tensor *Linear(Module *module, int layernum, Tensor *a, int in_features, int out_features, int use_bias);
#endif