#ifndef GHOSTFLOW_H
#define GHOSTFLOW_H

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * Error codes
 */
typedef enum GhostFlowError {
  Success = 0,
  InvalidShape = 1,
  InvalidData = 2,
  NullPointer = 3,
  AllocationFailed = 4,
  ComputationFailed = 5,
  Unknown = 99,
} GhostFlowError;

/**
 * Opaque handle to a GhostFlow tensor
 */
typedef struct GhostFlowTensor {
  uint8_t _private[0];
} GhostFlowTensor;

/**
 * Initialize GhostFlow library
 */
enum GhostFlowError ghostflow_init(void);

/**
 * Get GhostFlow version string
 */
const char *ghostflow_version(void);

/**
 * Free a version string
 */
void ghostflow_free_string(char *s);

/**
 * Create a new tensor from data
 *
 * # Arguments
 * * `data` - Pointer to float array
 * * `data_len` - Length of data array
 * * `shape` - Pointer to shape array
 * * `shape_len` - Length of shape array
 * * `out` - Output pointer to store the created tensor
 *
 * # Returns
 * Error code
 */
enum GhostFlowError ghostflow_tensor_create(const float *data,
                                            uintptr_t data_len,
                                            const uintptr_t *shape,
                                            uintptr_t shape_len,
                                            struct GhostFlowTensor **out);

/**
 * Create a tensor filled with zeros
 */
enum GhostFlowError ghostflow_tensor_zeros(const uintptr_t *shape,
                                           uintptr_t shape_len,
                                           struct GhostFlowTensor **out);

/**
 * Create a tensor filled with ones
 */
enum GhostFlowError ghostflow_tensor_ones(const uintptr_t *shape,
                                          uintptr_t shape_len,
                                          struct GhostFlowTensor **out);

/**
 * Free a tensor
 */
void ghostflow_tensor_free(struct GhostFlowTensor *tensor);

/**
 * Get tensor shape
 */
enum GhostFlowError ghostflow_tensor_shape(const struct GhostFlowTensor *tensor,
                                           uintptr_t *out_shape,
                                           uintptr_t *out_ndim);

/**
 * Get tensor data
 */
enum GhostFlowError ghostflow_tensor_data(const struct GhostFlowTensor *tensor,
                                          float *out_data,
                                          uintptr_t *out_len);

/**
 * Add two tensors
 */
enum GhostFlowError ghostflow_tensor_add(const struct GhostFlowTensor *a,
                                         const struct GhostFlowTensor *b,
                                         struct GhostFlowTensor **out);

/**
 * Multiply two tensors element-wise
 */
enum GhostFlowError ghostflow_tensor_mul(const struct GhostFlowTensor *a,
                                         const struct GhostFlowTensor *b,
                                         struct GhostFlowTensor **out);

/**
 * Matrix multiplication
 */
enum GhostFlowError ghostflow_tensor_matmul(const struct GhostFlowTensor *a,
                                            const struct GhostFlowTensor *b,
                                            struct GhostFlowTensor **out);

/**
 * Reshape a tensor
 */
enum GhostFlowError ghostflow_tensor_reshape(const struct GhostFlowTensor *tensor,
                                             const uintptr_t *new_shape,
                                             uintptr_t new_shape_len,
                                             struct GhostFlowTensor **out);

/**
 * Get last error message
 */
const char *ghostflow_get_last_error(void);

#endif /* GHOSTFLOW_H */
