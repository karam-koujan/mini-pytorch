#include <assert.h>
#include <stdio.h>
#include "../tensor.h"


void test_tensor_validate_shape() {
    printf("Running test cases for tensor_validate_shape...\n");

    // Test Case 1: Valid shapes for matrix multiplication
    Tensor a = tensor_empty(2, 2, 3,NULL,NULL,NULL); // Shape (2, 3)
    Tensor b = tensor_empty(2, 3, 4,NULL,NULL,NULL); // Shape (3, 4)
    assert(tensor_validate_shape(a, b) == 1);
    printf("Test Case 1 passed: Valid shapes.\n");

    // Test Case 2: Invalid shapes (number of dimensions mismatch)
    Tensor c = tensor_empty(3, 2, 3, 4,NULL,NULL,NULL); // Shape (2, 3, 4)
    Tensor d = tensor_empty(2, 3, 4,NULL,NULL,NULL);    // Shape (3, 4)
    assert(tensor_validate_shape(c, d) == -1);
    printf("Test Case 2 passed: Dimension mismatch.\n");

    // Test Case 3: Invalid shapes (inner dimensions mismatch)
    Tensor e = tensor_empty(2, 2, 3,NULL,NULL,NULL); // Shape (2, 3)
    Tensor f = tensor_empty(2, 4, 5,NULL,NULL,NULL); // Shape (4, 5)
    assert(tensor_validate_shape(e, f) == -1);
    printf("Test Case 3 passed: Inner dimensions mismatch.\n");

    // Test Case 4: Edge case (1D tensors, invalid for matrix multiplication)
    Tensor g = tensor_empty(1, 3,NULL,NULL,NULL); // Shape (3)
    Tensor h = tensor_empty(1, 3,NULL,NULL,NULL); // Shape (3)
    assert(tensor_validate_shape(g, h) == -1);
    printf("Test Case 4 passed: 1D tensors.\n");

    // Test Case 5: Large tensors with valid shapes
    Tensor i = tensor_empty(3, 10, 20, 30,NULL,NULL,NULL); // Shape (10, 20, 30)
    Tensor j = tensor_empty(3, 30, 40, 50,NULL,NULL,NULL); // Shape (30, 40, 50)
    assert(tensor_validate_shape(i, j) == -1);
    printf("Test Case 5 passed: Large tensors with valid shapes.\n");
    printf("All test cases passed!\n");
}

int main() {
    test_tensor_validate_shape();
    return 0;
}