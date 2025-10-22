#ifndef TENSOR_H
#define TENSOR_H

#include <stdarg.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#ifndef int64_t
#define int64_t long long
#endif
typedef enum e_type
{
	INT32,
	INT64,
	FLOAT32,
	DOUBLE
} Dtype;

typedef enum e_device
{
	CPU,
	GPU
} Device;


typedef	struct
{
	void *data;
	int64_t *shape;
	int64_t *strides;
	Dtype dtype;
	Device device;
	void *grad;
	int size;
	int	requires_grad;
	int num_dims;
	int	is_broadcasted;
	int64_t *prebroadcast_shape;
	int64_t *prebroadcast_stride;
	int	prebroadcast_dims;
	int	is_leaf;
	void *grad_fn;
} Tensor;

typedef struct Node
{
	Tensor *grad;
	Tensor **saved_tensors;
	int64_t *broadcasted_shape_a;
	int64_t *broadcasted_shape_b;
	Tensor **(*calculate_gradient)(struct Node *node,Tensor *grad);
}	Grad_Node;

typedef struct
{
	Tensor **parameters;
} Module;

int 	sizeof_type(Dtype type);
int64_t calculate_size(const int64_t *shape, int64_t ndim);
void 	tensor_set_seed(unsigned int seed);
float	generate_random();
int64_t *create_shape(const int64_t *shape, int64_t ndim);
int64_t *create_stride(const int64_t *shape, int64_t ndim, Dtype type);
void    *create_zero_data(Dtype type, int size);
void    *create_val_data(Dtype type, int size, void *val);
void    *create_one_data(Dtype type, int size);
void 	*copy_arr_data(void *arr, Dtype type, int64_t size);
void    *create_rand_data(Dtype type, int size);
Tensor *tensor_zeros(const int64_t *shape, int64_t ndim, Dtype type, Device device);
Tensor *tensor_ones(const int64_t *shape, int64_t ndim, Dtype type, Device device);
Tensor *tensor_full(const int64_t *shape, int64_t ndim, Dtype type, Device device, void *val);
Tensor *tensor_from_arr(void *arr, const int64_t *shape, int64_t ndim, Dtype type, Device device);
Tensor *tensor_rand(const int64_t *shape, int64_t ndim, Dtype type, Device device);
void 	tensor_infos(Tensor *tensor);
Tensor	*tensor_scalar(void *nb, Dtype type, Device device);
int		tensor_broadcast(Tensor *a, Tensor *b);
void    tensor_free(Tensor *a);
Tensor  *tensor_view(Tensor *a, const int64_t *view, int64_t new_ndim);
int		is_view_allowed(const int64_t *new_view, int64_t new_ndim);
int64_t *infer_shape_from_view(Tensor *a, const int64_t *view, int64_t new_ndim);
Tensor  *tensor_reshape(Tensor *a, const int64_t *view, int64_t new_ndim);
int is_contigious(Tensor *a);
Tensor *tensor_copy(Tensor *a);
Tensor *tensor_transpose(Tensor *a, int64_t dim0, int64_t dim1);
Tensor *tensor_t(Tensor *a);
Tensor *tensor_permute(Tensor *a, int64_t *dims, int64_t num_dims);
void    pairwise_op(Tensor *a, Tensor *b, Tensor *r, char op);
void    pairwise_add(void *a, void *b, Tensor *r);
Dtype 	promote_dtype(Dtype a, Dtype b);
void   *promote_data(Tensor *a, Dtype dtype);
void 	tensor_unbroadcast(Tensor *a);
Tensor *tensor_div(Tensor*a, Tensor *b);
Tensor *tensor_sub(Tensor*a, Tensor *b);
Tensor *tensor_add(Tensor*a, Tensor *b);
Tensor *tensor_mul(Tensor*a, Tensor *b);
Tensor *tensor_deep_copy(Tensor *a);
Tensor *tensor_matmul(Tensor*a, Tensor *b);
Tensor *tensor_mm(Tensor*a, Tensor *b);
int tensor_matmul_broadcast(Tensor *a, Tensor *b);
int is_tensor_broadcastable(Tensor *a, Tensor *b, int mat_p);
int64_t tensor_batchsize(int64_t *shape, int64_t dim);
void    fill_data(void *data, int offset, Dtype dtype, void *value);
void    *copy_contigious_data(Tensor *a, int size);
int tensor_contigous_broadcast(Tensor *a);
void matmul_calculation(Tensor *a_r, Tensor *b_r, Tensor *result);
void    tensor_free_after_reshape(Tensor *a);
Tensor *tensor_constructor(const int64_t *shape, int ndim, Dtype type, Device device);
Grad_Node	*create_matmul_node(Tensor *a, Tensor *b);
Grad_Node	*create_mm_node(Tensor *a, Tensor *b);
Grad_Node	*create_pairwise_mul_node(Tensor *a, Tensor *b);
Grad_Node	*create_add_node(Tensor *a, Tensor *b);
Grad_Node	*create_sub_node(Tensor *a, Tensor *b);
void	tensor_backward(Tensor *a, Tensor *prev_grad);
void tensor_set_require_grad(Tensor *a, int requires_grad);
// autograd flags
void tensor_set_require_grad(Tensor *a, int requires_grad);

// broadcasting collapse helper
Tensor *tensor_collapse(Tensor *a, Tensor *b);

// grad node creators
Grad_Node *create_matmul_node(Tensor *a, Tensor *b);
Grad_Node *create_mm_node(Tensor *a, Tensor *b);
Grad_Node *create_pairwise_mul_node(Tensor *a, Tensor *b);
Grad_Node *create_add_node(Tensor *a, Tensor *b);
Grad_Node *create_sub_node(Tensor *a, Tensor *b);

// backward functions
Tensor **tensor_backadd(Grad_Node *node, Tensor *grad);
Tensor **tensor_backsub(Grad_Node *node, Tensor *grad);
Tensor **tensor_backmatmul(Grad_Node *node, Tensor *grad);
Tensor **tensor_backpairwise_mul(Grad_Node *node, Tensor *grad);
Tensor **tensor_backmm(Grad_Node *node, Tensor *grad);

// grad accumulation + traversal
void tensor_accumulate_grad(Tensor *a, Tensor *grad);
void tensor_backward(Tensor *a, Tensor *prev_grad);
int is_shape_allowed(Tensor*a, Tensor *b);
#endif