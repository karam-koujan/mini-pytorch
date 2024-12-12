#include "test.h"


void test_tensor_empty() {
    Tensor *tensor = tensor_empty(2, 3, 4,0); // Create a 2D tensor (3x4)
    assert(tensor->num_dims == 2);
    assert(tensor->shape[0] == 3);
    assert(tensor->shape[1] == 4);
    assert(tensor->data != NULL);
    printf("test_tensor_empty passed.\n");

    free(tensor->data);
    free(tensor->shape);
    free(tensor->strides);
}

void test_tensor_ones() {
    Tensor *tensor = tensor_ones(2, 3, 4,0); // Create a 2D tensor (3x4) filled with ones
    float *data = tensor->data;

    for (int i = 0; i < tensor_entries_len(tensor); i++) {
        assert(data[i] == 1.0f);
    }
    printf("test_tensor_ones passed.\n");

    free(tensor->data);
    free(tensor->shape);
    free(tensor->strides);
}

void test_tensor_rand() {
    tensor_set_seed(42); // Set seed for reproducibility
    Tensor *tensor = tensor_rand(2, 3, 4,0); // Create a 2D tensor (3x4) with random values
    float *data = tensor->data;

    for (int i = 0; i < tensor_entries_len(tensor); i++) {
        assert(data[i] >= 0.0f && data[i] < 1.0f);
    }
    printf("test_tensor_rand passed.\n");

    free(tensor->data);
    free(tensor->shape);
    free(tensor->strides);
}

void test_tensor_entries_len() {
    Tensor *tensor = tensor_empty(3, 2, 3, 4,0); // Create a 3D tensor (2x3x4)
    int entries = tensor_entries_len(tensor);
    assert(entries == 2 * 3 * 4);
    printf("test_tensor_entries_len passed.\n");

    free(tensor->data);
    free(tensor->shape);
    free(tensor->strides);
}


void test_tensor_fill() {
    Tensor *tensor = tensor_empty(2, 3, 4,0); // Create an empty 2D tensor (3x4)
    tensor_fill(tensor, 5.0f); // Fill it with the value 5.0

    float *data = tensor->data;
    for (int i = 0; i < tensor_entries_len(tensor); i++) {
        assert(data[i] == 5.0f);
    }
    printf("test_tensor_fill passed.\n");

    free(tensor->data);
    free(tensor->shape);
    free(tensor->strides);
}

void test_tensor_full() {
    Tensor *tensor = tensor_full(2, 3, 4, 7.0f,0); // Create a 2D tensor (3x4) filled with 7.0
    float *data = tensor->data;

    for (int i = 0; i < tensor_entries_len(tensor); i++) {
        assert(data[i] == 7.0f);
    }
    printf("test_tensor_full passed.\n");

    free(tensor->data);
    free(tensor->shape);
    free(tensor->strides);
}


void test_add_options() {
    Tensor *tensor = tensor_empty(2, 3, 4,2, "gpu", "int");
    assert(tensor->device == GPU);
    assert(tensor->dtype == INT32);
    printf("test_add_options passed.\n");

    free(tensor->data);
    free(tensor->shape);
    free(tensor->strides);
}
void test_tensor_zeros() {
    Tensor *tensor = tensor_zeros(2, 3, 4,0); // Create a 2D tensor (3x4) filled with zeros
    float *data = tensor->data;

    for (int i = 0; i < tensor_entries_len(tensor); i++) {
        assert(data[i] == 0.0f);
    }
    printf("test_tensor_zeros passed.\n");

    free(tensor->data);
    free(tensor->shape);
    free(tensor->strides);
}


