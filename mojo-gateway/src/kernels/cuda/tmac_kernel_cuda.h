/**
 * EdgeLLM CUDA T-MAC Kernel Header
 *
 * GPU-accelerated lookup table-based inference for BitNet 1.58-bit models.
 * Uses CUDA for parallel computation on NVIDIA GPUs.
 *
 * Target: NVIDIA Jetson Nano/Orin, RTX GPUs
 */

#ifndef TMAC_KERNEL_CUDA_H
#define TMAC_KERNEL_CUDA_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize CUDA context and allocate device memory.
 *
 * Must be called before any CUDA operations.
 *
 * @param max_weights_bytes   Maximum size of weight buffer in bytes
 * @param max_activations     Maximum number of activation elements
 * @param max_output          Maximum number of output elements
 * @return 0 on success, -1 on failure
 */
int cuda_init(int max_weights_bytes, int max_activations, int max_output);

/**
 * Cleanup CUDA resources.
 *
 * Frees all allocated device memory.
 */
void cuda_cleanup(void);

/**
 * T-MAC Matrix Multiplication (CUDA)
 *
 * GPU-accelerated BitNet inference using table lookups.
 * Each weight byte encodes 4 ternary values {-1, 0, +1}.
 *
 * @param weights     Packed ternary weights (host memory) [M * (K/4)]
 * @param activations Input activations (host memory) [K * N]
 * @param output      Output buffer (host memory) [M * N]
 * @param scales      Per-row scaling factors (host memory) [M]
 * @param M           Number of output rows
 * @param N           Number of output columns (batch size)
 * @param K           Inner dimension
 * @return 0 on success, -1 on failure
 */
int tmac_matmul_cuda(
    const int8_t* weights,
    const float* activations,
    float* output,
    const float* scales,
    int M, int N, int K
);

/**
 * RMSNorm (CUDA)
 *
 * GPU-accelerated Root Mean Square Layer Normalization.
 * output[i] = (input[i] / rms) * weight[i]
 * where rms = sqrt(mean(input^2) + eps)
 *
 * @param output      Output buffer (host memory) [batch_size * size]
 * @param input       Input buffer (host memory) [batch_size * size]
 * @param weight      Weight buffer (host memory) [size]
 * @param batch_size  Number of batches
 * @param size        Vector size per batch
 * @param eps         Epsilon for numerical stability
 * @return 0 on success, -1 on failure
 */
int rmsnorm_cuda(
    float* output,
    const float* input,
    const float* weight,
    int batch_size,
    int size,
    float eps
);

/**
 * Softmax (CUDA)
 *
 * GPU-accelerated numerically stable softmax.
 * softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
 *
 * @param output      Output buffer (host memory) [batch_size * size]
 * @param input       Input buffer (host memory) [batch_size * size]
 * @param batch_size  Number of batches
 * @param size        Vector size per batch
 * @return 0 on success, -1 on failure
 */
int softmax_cuda(
    float* output,
    const float* input,
    int batch_size,
    int size
);

/**
 * Check if CUDA is available.
 *
 * @return 1 if CUDA device is available, 0 otherwise
 */
int cuda_available(void);

/**
 * Get CUDA device name.
 *
 * @return Device name string (static buffer, do not free)
 */
const char* cuda_device_name(void);

/**
 * Synchronize CUDA device.
 *
 * Blocks until all CUDA operations complete.
 */
void cuda_sync(void);

/**
 * Get CUDA device properties.
 *
 * @param total_memory    Output: Total device memory in bytes (can be NULL)
 * @param sm_count        Output: Number of streaming multiprocessors (can be NULL)
 * @param compute_major   Output: Compute capability major version (can be NULL)
 * @param compute_minor   Output: Compute capability minor version (can be NULL)
 * @return 0 on success, -1 on failure
 */
int cuda_device_info(
    size_t* total_memory,
    int* sm_count,
    int* compute_major,
    int* compute_minor
);

#ifdef __cplusplus
}
#endif

#endif // TMAC_KERNEL_CUDA_H
