/**
 * cuBLAS Matrix Multiplication Kernels for EdgeLLM
 * Header file for FFI integration with Mojo
 */

#ifndef CUBLAS_MATMUL_H
#define CUBLAS_MATMUL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialization
int cublas_init(size_t weight_bytes, size_t activation_bytes);
int cublas_upload_weights(const float* weights_cpu, size_t bytes);
void cublas_cleanup();
void cublas_sync();

// Memory access
float* get_weights_gpu();
float* get_activations_gpu();

// Core matmul operations
int cublas_matvec(float* out_gpu, const float* x_gpu, const float* W_gpu, int out_dim, int in_dim);
int cublas_matvec_batched(float* out_gpu, const float* x_gpu, const float* W_gpu, int batch, int out_dim, int in_dim);
int cublas_add_bias(float* out_gpu, const float* bias_gpu, int size);

// Layer operations
int gpu_rmsnorm(float* out_gpu, const float* x_gpu, const float* weight_gpu, int size, float eps);
int gpu_swiglu(float* out_gpu, const float* gate_gpu, const float* up_gpu, int size);
int gpu_residual_add(float* x_gpu, const float* residual_gpu, int size);
int gpu_rope(float* q_gpu, float* k_gpu, const float* cos_gpu, const float* sin_gpu, int n_heads, int n_kv_heads, int head_dim);

// GQA Attention
int gpu_gqa_attention(float* output_gpu, const float* Q_gpu, const float* K_cache_gpu, const float* V_cache_gpu,
                      int n_heads, int n_kv_heads, int seq_len, int max_seq, int head_dim);
int gpu_kv_cache_update(float* K_cache_gpu, float* V_cache_gpu, const float* K_gpu, const float* V_gpu,
                        int n_kv_heads, int pos, int max_seq, int head_dim);

// Sampling
int gpu_argmax(int* result_gpu, const float* logits_gpu, int size);

// CUDA memory operations (for FFI)
int cuda_memcpy_d2d(float* dst, const float* src, size_t bytes);
int cuda_memcpy_d2h(float* dst_host, const float* src_device, size_t bytes);
int cuda_memcpy_h2d(float* dst_device, const float* src_host, size_t bytes);

// High-level inference API (recommended for FFI)
int gpu_configure(int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads,
                  int vocab_size, int seq_len, int has_bias);
int gpu_forward(int token, int pos);

#ifdef __cplusplus
}
#endif

#endif // CUBLAS_MATMUL_H
