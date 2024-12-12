#include "test.h"

int main() {
    test_tensor_empty();
    test_tensor_zeros();
    test_tensor_ones();
    test_tensor_rand();
    test_tensor_fill();
    test_tensor_full();
    test_tensor_entries_len();
    test_add_options();
	
    printf("All tests passed successfully!\n");
    return 0;
}
