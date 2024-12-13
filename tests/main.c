#include "test.h"

int main() {
    // test_tensor_empty();
    // test_tensor_zeros();
    // test_tensor_ones();
    // test_tensor_rand();
    // test_tensor_fill();
    // test_tensor_full();
    // test_tensor_entries_len();
    // test_add_options();
	// test_tensor_validate_shape();	
	// test_tensor_is_broadcastable();
    // printf("All tests passed successfully!\n");
	Tensor *a = tensor_rand(2,2,3,0);
	int shape[3] = {3,2};
	tensor_print(a);
	Tensor *at = tensor_reshape(a,2,shape);
	tensor_print(at);
	printf("%f",tensor_get_num(at,1,1));
    return 0;
}
