#ifndef TEST_H
#define TEST_H

#include <assert.h>
#include <stdio.h>
#include "../tensor.h"

void test_tensor_empty();
void test_tensor_ones();
void test_tensor_rand();
void test_tensor_entries_len();
void test_tensor_fill();
void test_tensor_full();
void test_add_options();
void test_tensor_zeros();
void test_tensor_validate_shape();
void test_tensor_is_broadcastable();

#endif