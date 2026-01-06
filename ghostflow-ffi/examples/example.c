/**
 * GhostFlow C FFI Example
 * 
 * This example demonstrates how to use GhostFlow from C.
 * 
 * Compile:
 *   gcc example.c -L../target/release -lghostflow_ffi -o example
 * 
 * Run:
 *   LD_LIBRARY_PATH=../target/release ./example
 */

#include <stdio.h>
#include <stdlib.h>
#include "../ghostflow.h"

int main() {
    printf("=== GhostFlow C FFI Example ===\n\n");

    // Initialize GhostFlow
    GhostFlowError err = ghostflow_init();
    if (err != Success) {
        printf("Failed to initialize GhostFlow\n");
        return 1;
    }

    // Get version
    const char* version = ghostflow_version();
    printf("GhostFlow version: %s\n\n", version);

    // 1. Create a tensor
    printf("1. Creating tensor...\n");
    float data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    size_t shape[] = {2, 3};
    GhostFlowTensor* tensor = NULL;

    err = ghostflow_tensor_create(data, 6, shape, 2, &tensor);
    if (err != Success) {
        printf("Failed to create tensor\n");
        return 1;
    }
    printf("✓ Tensor created: shape [2, 3]\n\n");

    // 2. Create zeros tensor
    printf("2. Creating zeros tensor...\n");
    size_t zeros_shape[] = {2, 2};
    GhostFlowTensor* zeros = NULL;

    err = ghostflow_tensor_zeros(zeros_shape, 2, &zeros);
    if (err != Success) {
        printf("Failed to create zeros tensor\n");
        return 1;
    }
    printf("✓ Zeros tensor created: shape [2, 2]\n\n");

    // 3. Create ones tensor
    printf("3. Creating ones tensor...\n");
    size_t ones_shape[] = {2, 2};
    GhostFlowTensor* ones = NULL;

    err = ghostflow_tensor_ones(ones_shape, 2, &ones);
    if (err != Success) {
        printf("Failed to create ones tensor\n");
        return 1;
    }
    printf("✓ Ones tensor created: shape [2, 2]\n\n");

    // 4. Add tensors
    printf("4. Adding tensors...\n");
    GhostFlowTensor* sum = NULL;

    err = ghostflow_tensor_add(zeros, ones, &sum);
    if (err != Success) {
        printf("Failed to add tensors\n");
        return 1;
    }
    printf("✓ Tensors added successfully\n\n");

    // 5. Matrix multiplication
    printf("5. Matrix multiplication...\n");
    float mat_a[] = {1.0, 2.0, 3.0, 4.0};
    float mat_b[] = {5.0, 6.0, 7.0, 8.0};
    size_t mat_shape[] = {2, 2};
    
    GhostFlowTensor* a = NULL;
    GhostFlowTensor* b = NULL;
    GhostFlowTensor* result = NULL;

    ghostflow_tensor_create(mat_a, 4, mat_shape, 2, &a);
    ghostflow_tensor_create(mat_b, 4, mat_shape, 2, &b);
    
    err = ghostflow_tensor_matmul(a, b, &result);
    if (err != Success) {
        printf("Failed to multiply matrices\n");
        return 1;
    }
    printf("✓ Matrix multiplication successful\n\n");

    // 6. Get result data
    printf("6. Reading result data...\n");
    float result_data[4];
    size_t result_len;

    err = ghostflow_tensor_data(result, result_data, &result_len);
    if (err != Success) {
        printf("Failed to read tensor data\n");
        return 1;
    }

    printf("Result: [");
    for (size_t i = 0; i < result_len; i++) {
        printf("%.1f", result_data[i]);
        if (i < result_len - 1) printf(", ");
    }
    printf("]\n\n");

    // Cleanup
    printf("7. Cleaning up...\n");
    ghostflow_tensor_free(tensor);
    ghostflow_tensor_free(zeros);
    ghostflow_tensor_free(ones);
    ghostflow_tensor_free(sum);
    ghostflow_tensor_free(a);
    ghostflow_tensor_free(b);
    ghostflow_tensor_free(result);
    ghostflow_free_string((char*)version);
    printf("✓ All resources freed\n\n");

    printf("=== Example completed successfully! ===\n");
    return 0;
}
