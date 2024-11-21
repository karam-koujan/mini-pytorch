
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

struct Tensor
{
	void *data;
	int *shape;
	int *strides;
	enum Dtype dtype;
	enum Device device;
	void *grad;
	int	requires_grad;
	int num_dims;
};
