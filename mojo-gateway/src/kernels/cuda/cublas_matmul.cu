/**
 * cuBLAS Matrix Multiplication Kernels for EdgeLLM
 *
 * This is the KEY to 400+ tok/s performance.
 * Matmuls are 90%+ of LLM inference compute.
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

// Global cuBLAS handle - reuse across calls
static cublasHandle_t g_cublas_handle = nullptr;
static cudaStream_t g_stream = nullptr;

// Persistent GPU buffers for weights and activations
static float* g_weights_gpu = nullptr;
static size_t g_weights_size = 0;
static float* g_act_gpu = nullptr;
static size_t g_act_size = 0;

extern "C" {

/**
 * Initialize cuBLAS and allocate GPU memory for model weights.
 * Call once at startup.
 */
int cublas_init(size_t weight_bytes, size_t activation_bytes) {
    cublasStatus_t status = cublasCreate(&g_cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS init failed: %d\n", status);
        return -1;
    }

    cudaStreamCreate(&g_stream);
    cublasSetStream(g_cublas_handle, g_stream);

    // Use Tensor Cores when available (TF32 on Ampere+)
    cublasSetMathMode(g_cublas_handle, CUBLAS_DEFAULT_MATH);

    // Allocate GPU memory for weights (loaded once)
    if (weight_bytes > 0) {
        cudaMalloc(&g_weights_gpu, weight_bytes);
        g_weights_size = weight_bytes;
        printf("Allocated %.2f GB for weights on GPU\n", weight_bytes / 1e9);
    }

    // Allocate GPU memory for activations (reused each forward pass)
    if (activation_bytes > 0) {
        cudaMalloc(&g_act_gpu, activation_bytes);
        g_act_size = activation_bytes;
        printf("Allocated %.2f MB for activations on GPU\n", activation_bytes / 1e6);
    }

    return 0;
}

/**
 * Upload model weights to GPU (call once after loading model).
 */
int cublas_upload_weights(const float* weights_cpu, size_t bytes) {
    if (!g_weights_gpu || bytes > g_weights_size) {
        printf("Weight buffer not allocated or too small\n");
        return -1;
    }

    cudaError_t err = cudaMemcpy(g_weights_gpu, weights_cpu, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Weight upload failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    printf("Uploaded %.2f GB weights to GPU\n", bytes / 1e9);
    return 0;
}

/**
 * Matrix-vector multiplication: out = W @ x
 * W is [out_dim, in_dim], x is [in_dim], out is [out_dim]
 *
 * This is the core operation - called millions of times per second.
 */
int cublas_matvec(
    float* out_gpu,      // Output [out_dim]
    const float* x_gpu,  // Input [in_dim]
    const float* W_gpu,  // Weight [out_dim, in_dim]
    int out_dim,
    int in_dim
) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // GEMV: y = alpha * A * x + beta * y
    // A is out_dim x in_dim (row-major in C, so we use CUBLAS_OP_T)
    cublasStatus_t status = cublasSgemv(
        g_cublas_handle,
        CUBLAS_OP_T,        // Transpose because row-major
        in_dim,             // rows of A
        out_dim,            // cols of A
        &alpha,
        W_gpu, in_dim,      // A, lda
        x_gpu, 1,           // x, incx
        &beta,
        out_gpu, 1          // y, incy
    );

    return (status == CUBLAS_STATUS_SUCCESS) ? 0 : -1;
}

/**
 * Batched matrix multiplication for all layers at once.
 * Uses strided batched GEMM for maximum throughput.
 */
int cublas_matvec_batched(
    float* out_gpu,      // Output [batch, out_dim]
    const float* x_gpu,  // Input [batch, in_dim]
    const float* W_gpu,  // Weight [batch, out_dim, in_dim]
    int batch,           // Number of layers
    int out_dim,
    int in_dim
) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    long long strideA = (long long)out_dim * in_dim;
    long long strideB = in_dim;
    long long strideC = out_dim;

    // Strided batched GEMV (treated as GEMM with n=1)
    cublasStatus_t status = cublasSgemmStridedBatched(
        g_cublas_handle,
        CUBLAS_OP_T,        // A transposed (row-major)
        CUBLAS_OP_N,        // B not transposed
        out_dim,            // m
        1,                  // n (single vector)
        in_dim,             // k
        &alpha,
        W_gpu, in_dim, strideA,  // A, lda, strideA
        x_gpu, in_dim, strideB,  // B, ldb, strideB
        &beta,
        out_gpu, out_dim, strideC,  // C, ldc, strideC
        batch
    );

    return (status == CUBLAS_STATUS_SUCCESS) ? 0 : -1;
}

/**
 * Add bias to output vector (in-place).
 */
__global__ void add_bias_kernel(float* out, const float* bias, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] += bias[idx];
    }
}

int cublas_add_bias(float* out_gpu, const float* bias_gpu, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    add_bias_kernel<<<blocks, threads, 0, g_stream>>>(out_gpu, bias_gpu, size);
    return 0;
}

/**
 * RMSNorm on GPU with proper block reduction.
 */
__global__ void rmsnorm_kernel(
    float* out,
    const float* x,
    const float* weight,
    int size,
    float eps
) {
    __shared__ float s_partial[32];  // For warp results

    // Compute local sum of squares
    float local_ss = 0.0f;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        float v = x[i];
        local_ss += v * v;
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        local_ss += __shfl_down_sync(0xffffffff, local_ss, offset);
    }

    // Store warp results
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) {
        s_partial[warp_id] = local_ss;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        local_ss = (lane_id < blockDim.x / 32) ? s_partial[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            local_ss += __shfl_down_sync(0xffffffff, local_ss, offset);
        }
        if (lane_id == 0) {
            s_partial[0] = rsqrtf(local_ss / size + eps);
        }
    }
    __syncthreads();

    // Normalize and scale
    float scale = s_partial[0];
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        out[i] = weight[i] * x[i] * scale;
    }
}

int gpu_rmsnorm(
    float* out_gpu,
    const float* x_gpu,
    const float* weight_gpu,
    int size,
    float eps
) {
    rmsnorm_kernel<<<1, 256, 0, g_stream>>>(out_gpu, x_gpu, weight_gpu, size, eps);
    return 0;
}

/**
 * SwiGLU activation: out = silu(gate) * up
 */
__global__ void swiglu_kernel(float* out, const float* gate, const float* up, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = gate[idx];
        float silu_g = g / (1.0f + expf(-g));  // SiLU = x * sigmoid(x)
        out[idx] = silu_g * up[idx];
    }
}

int gpu_swiglu(float* out_gpu, const float* gate_gpu, const float* up_gpu, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    swiglu_kernel<<<blocks, threads, 0, g_stream>>>(out_gpu, gate_gpu, up_gpu, size);
    return 0;
}

/**
 * Residual add: x += residual
 */
__global__ void residual_add_kernel(float* x, const float* residual, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] += residual[idx];
    }
}

int gpu_residual_add(float* x_gpu, const float* residual_gpu, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    residual_add_kernel<<<blocks, threads, 0, g_stream>>>(x_gpu, residual_gpu, size);
    return 0;
}

/**
 * RoPE (Rotary Position Embedding) on GPU.
 */
__global__ void rope_kernel(
    float* q,           // [n_heads, head_dim]
    float* k,           // [n_kv_heads, head_dim]
    const float* cos,   // [head_dim/2]
    const float* sin,   // [head_dim/2]
    int n_heads,
    int n_kv_heads,
    int head_dim
) {
    int head = blockIdx.x;
    int j = threadIdx.x * 2;  // Process pairs

    if (j >= head_dim) return;

    float c = cos[j / 2];
    float s = sin[j / 2];

    // Apply RoPE to Q
    if (head < n_heads) {
        int idx = head * head_dim + j;
        float q0 = q[idx];
        float q1 = q[idx + 1];
        q[idx] = q0 * c - q1 * s;
        q[idx + 1] = q0 * s + q1 * c;
    }

    // Apply RoPE to K (fewer heads)
    if (head < n_kv_heads) {
        int idx = head * head_dim + j;
        float k0 = k[idx];
        float k1 = k[idx + 1];
        k[idx] = k0 * c - k1 * s;
        k[idx + 1] = k0 * s + k1 * c;
    }
}

int gpu_rope(
    float* q_gpu,
    float* k_gpu,
    const float* cos_gpu,
    const float* sin_gpu,
    int n_heads,
    int n_kv_heads,
    int head_dim
) {
    int max_heads = (n_heads > n_kv_heads) ? n_heads : n_kv_heads;
    rope_kernel<<<max_heads, head_dim / 2, 0, g_stream>>>(
        q_gpu, k_gpu, cos_gpu, sin_gpu, n_heads, n_kv_heads, head_dim
    );
    return 0;
}

/**
 * GQA (Grouped Query Attention) decode kernel with proper block reduction.
 * This handles n_heads != n_kv_heads properly.
 *
 * For Qwen 1.5B: 12 Q heads, 2 KV heads, kv_mul = 6
 */
__global__ void gqa_attention_kernel(
    float* output,           // [n_heads, head_dim]
    const float* Q,          // [n_heads, head_dim]
    const float* K_cache,    // [n_kv_heads, max_seq, head_dim]
    const float* V_cache,    // [n_kv_heads, max_seq, head_dim]
    int n_heads,
    int n_kv_heads,
    int seq_len,             // Current sequence length (pos + 1)
    int max_seq,             // Maximum sequence length (stride)
    int head_dim,
    float scale
) {
    int q_head = blockIdx.x;
    int kv_head = q_head / (n_heads / n_kv_heads);  // Which KV head this Q uses

    extern __shared__ float smem[];
    float* s_scores = smem;                         // [seq_len]
    float* s_partial = smem + seq_len;              // [32] for warp results

    // 1. Compute attention scores: Q @ K^T
    float local_max = -1e10f;
    for (int t = threadIdx.x; t < seq_len; t += blockDim.x) {
        float score = 0.0f;
        int k_offset = kv_head * max_seq * head_dim + t * head_dim;
        int q_offset = q_head * head_dim;

        for (int d = 0; d < head_dim; d++) {
            score += Q[q_offset + d] * K_cache[k_offset + d];
        }
        score *= scale;
        s_scores[t] = score;
        local_max = fmaxf(local_max, score);
    }

    // Block reduction for max
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    for (int offset = 16; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    if (lane_id == 0) s_partial[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0) {
        local_max = (lane_id < blockDim.x / 32) ? s_partial[lane_id] : -1e10f;
        for (int offset = 16; offset > 0; offset /= 2) {
            local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
        }
        if (lane_id == 0) s_partial[0] = local_max;
    }
    __syncthreads();
    float max_score = s_partial[0];

    // 2. Softmax: exp and sum
    float local_sum = 0.0f;
    for (int t = threadIdx.x; t < seq_len; t += blockDim.x) {
        float exp_score = expf(s_scores[t] - max_score);
        s_scores[t] = exp_score;
        local_sum += exp_score;
    }

    // Block reduction for sum
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    if (lane_id == 0) s_partial[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        local_sum = (lane_id < blockDim.x / 32) ? s_partial[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        }
        if (lane_id == 0) s_partial[0] = local_sum;
    }
    __syncthreads();

    float inv_sum = 1.0f / s_partial[0];
    for (int t = threadIdx.x; t < seq_len; t += blockDim.x) {
        s_scores[t] *= inv_sum;
    }
    __syncthreads();

    // 3. Weighted sum of V
    int out_offset = q_head * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            int v_offset = kv_head * max_seq * head_dim + t * head_dim + d;
            acc += s_scores[t] * V_cache[v_offset];
        }
        output[out_offset + d] = acc;
    }
}

int gpu_gqa_attention(
    float* output_gpu,
    const float* Q_gpu,
    const float* K_cache_gpu,
    const float* V_cache_gpu,
    int n_heads,
    int n_kv_heads,
    int seq_len,
    int max_seq,
    int head_dim
) {
    float scale = 1.0f / sqrtf((float)head_dim);
    int smem_size = seq_len * sizeof(float);

    // One block per Q head
    gqa_attention_kernel<<<n_heads, 128, smem_size, g_stream>>>(
        output_gpu, Q_gpu, K_cache_gpu, V_cache_gpu,
        n_heads, n_kv_heads, seq_len, max_seq, head_dim, scale
    );

    return 0;
}

/**
 * Copy current K, V to cache.
 */
__global__ void kv_cache_update_kernel(
    float* K_cache,      // [n_kv_heads, max_seq, head_dim]
    float* V_cache,
    const float* K,      // [n_kv_heads, head_dim]
    const float* V,
    int n_kv_heads,
    int pos,
    int max_seq,
    int head_dim
) {
    int head = blockIdx.x;
    int d = threadIdx.x;

    if (head < n_kv_heads && d < head_dim) {
        int cache_idx = head * max_seq * head_dim + pos * head_dim + d;
        int src_idx = head * head_dim + d;
        K_cache[cache_idx] = K[src_idx];
        V_cache[cache_idx] = V[src_idx];
    }
}

int gpu_kv_cache_update(
    float* K_cache_gpu,
    float* V_cache_gpu,
    const float* K_gpu,
    const float* V_gpu,
    int n_kv_heads,
    int pos,
    int max_seq,
    int head_dim
) {
    kv_cache_update_kernel<<<n_kv_heads, head_dim, 0, g_stream>>>(
        K_cache_gpu, V_cache_gpu, K_gpu, V_gpu,
        n_kv_heads, pos, max_seq, head_dim
    );
    return 0;
}

/**
 * Argmax for greedy sampling.
 */
__global__ void argmax_kernel(int* result, const float* logits, int size) {
    __shared__ float s_max_val[256];
    __shared__ int s_max_idx[256];

    int tid = threadIdx.x;
    float max_val = -1e10f;
    int max_idx = 0;

    for (int i = tid; i < size; i += blockDim.x) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }

    s_max_val[tid] = max_val;
    s_max_idx[tid] = max_idx;
    __syncthreads();

    // Reduce
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s && s_max_val[tid + s] > s_max_val[tid]) {
            s_max_val[tid] = s_max_val[tid + s];
            s_max_idx[tid] = s_max_idx[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *result = s_max_idx[0];
    }
}

int gpu_argmax(int* result_gpu, const float* logits_gpu, int size) {
    argmax_kernel<<<1, 256, 0, g_stream>>>(result_gpu, logits_gpu, size);
    return 0;
}

/**
 * Synchronize stream.
 */
void cublas_sync() {
    cudaStreamSynchronize(g_stream);
}

/**
 * Cleanup.
 */
void cublas_cleanup() {
    if (g_weights_gpu) cudaFree(g_weights_gpu);
    if (g_act_gpu) cudaFree(g_act_gpu);
    if (g_stream) cudaStreamDestroy(g_stream);
    if (g_cublas_handle) cublasDestroy(g_cublas_handle);
    g_weights_gpu = nullptr;
    g_act_gpu = nullptr;
    g_stream = nullptr;
    g_cublas_handle = nullptr;
}

/**
 * Get pointers to GPU buffers.
 */
float* get_weights_gpu() { return g_weights_gpu; }
float* get_activations_gpu() { return g_act_gpu; }

/**
 * CUDA memory operations (wrappers for FFI).
 */
int cuda_memcpy_d2d(float* dst, const float* src, size_t bytes) {
    cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, g_stream);
    return (err == cudaSuccess) ? 0 : -1;
}

int cuda_memcpy_d2h(float* dst_host, const float* src_device, size_t bytes) {
    cudaStreamSynchronize(g_stream);  // Ensure all ops complete first
    cudaError_t err = cudaMemcpy(dst_host, src_device, bytes, cudaMemcpyDeviceToHost);
    return (err == cudaSuccess) ? 0 : -1;
}

int cuda_memcpy_h2d(float* dst_device, const float* src_host, size_t bytes) {
    cudaError_t err = cudaMemcpyAsync(dst_device, src_host, bytes, cudaMemcpyHostToDevice, g_stream);
    return (err == cudaSuccess) ? 0 : -1;
}

/**
 * Full transformer forward pass on GPU.
 * This is the main entry point - handles all memory management internally.
 */
// Model configuration (set once)
static int g_dim = 0;
static int g_hidden_dim = 0;
static int g_n_layers = 0;
static int g_n_heads = 0;
static int g_n_kv_heads = 0;
static int g_vocab_size = 0;
static int g_seq_len = 0;
static int g_head_dim = 0;
static int g_kv_dim = 0;
static bool g_has_bias = false;

// Weight offsets in global buffer
static size_t g_token_emb_offset = 0;
static size_t g_rms_att_offset = 0;
static size_t g_wq_offset = 0;
static size_t g_wk_offset = 0;
static size_t g_wv_offset = 0;
static size_t g_wo_offset = 0;
static size_t g_rms_ffn_offset = 0;
static size_t g_w1_offset = 0;
static size_t g_w2_offset = 0;
static size_t g_w3_offset = 0;
static size_t g_rms_final_offset = 0;
static size_t g_freq_cos_offset = 0;
static size_t g_freq_sin_offset = 0;
static size_t g_bq_offset = 0;
static size_t g_bk_offset = 0;
static size_t g_bv_offset = 0;

// Activation offsets in activation buffer
static size_t g_x_offset = 0;
static size_t g_xb_offset = 0;
static size_t g_xb2_offset = 0;
static size_t g_q_offset = 0;
static size_t g_k_offset = 0;
static size_t g_v_offset = 0;
static size_t g_hb_offset = 0;
static size_t g_hb2_offset = 0;
static size_t g_logits_offset = 0;
static size_t g_k_cache_offset = 0;
static size_t g_v_cache_offset = 0;
static size_t g_result_offset = 0;

/**
 * Configure model dimensions. Call once after loading model.
 */
int gpu_configure(int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads,
                  int vocab_size, int seq_len, int has_bias) {
    g_dim = dim;
    g_hidden_dim = hidden_dim;
    g_n_layers = n_layers;
    g_n_heads = n_heads;
    g_n_kv_heads = n_kv_heads;
    g_vocab_size = vocab_size;
    g_seq_len = seq_len;
    g_head_dim = dim / n_heads;
    g_kv_dim = (n_kv_heads * dim) / n_heads;
    g_has_bias = (has_bias != 0);

    // Calculate weight offsets
    size_t offset = 0;
    g_token_emb_offset = offset; offset += vocab_size * dim;
    g_rms_att_offset = offset; offset += n_layers * dim;
    g_wq_offset = offset; offset += n_layers * dim * dim;
    g_wk_offset = offset; offset += n_layers * g_kv_dim * dim;
    g_wv_offset = offset; offset += n_layers * g_kv_dim * dim;
    g_wo_offset = offset; offset += n_layers * dim * dim;
    g_rms_ffn_offset = offset; offset += n_layers * dim;
    g_w1_offset = offset; offset += n_layers * hidden_dim * dim;
    g_w2_offset = offset; offset += n_layers * dim * hidden_dim;
    g_w3_offset = offset; offset += n_layers * hidden_dim * dim;
    g_rms_final_offset = offset; offset += dim;
    g_freq_cos_offset = offset; offset += seq_len * (g_head_dim / 2);
    g_freq_sin_offset = offset; offset += seq_len * (g_head_dim / 2);

    if (has_bias) {
        g_bq_offset = offset; offset += n_layers * dim;
        g_bk_offset = offset; offset += n_layers * g_kv_dim;
        g_bv_offset = offset; offset += n_layers * g_kv_dim;
    }

    // Calculate activation offsets
    size_t act_offset = 0;
    g_x_offset = act_offset; act_offset += dim;
    g_xb_offset = act_offset; act_offset += dim;
    g_xb2_offset = act_offset; act_offset += dim;
    g_q_offset = act_offset; act_offset += dim;
    g_k_offset = act_offset; act_offset += g_kv_dim;
    g_v_offset = act_offset; act_offset += g_kv_dim;
    g_hb_offset = act_offset; act_offset += hidden_dim;
    g_hb2_offset = act_offset; act_offset += hidden_dim;
    g_logits_offset = act_offset; act_offset += vocab_size;
    g_k_cache_offset = act_offset; act_offset += n_layers * n_kv_heads * seq_len * g_head_dim;
    g_v_cache_offset = act_offset; act_offset += n_layers * n_kv_heads * seq_len * g_head_dim;
    g_result_offset = act_offset; act_offset += 1;

    printf("GPU configured: dim=%d, layers=%d, heads=%d/%d, vocab=%d, seq=%d\n",
           dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len);
    return 0;
}

/**
 * Full forward pass. Returns next token ID.
 */
int gpu_forward(int token, int pos) {
    if (!g_weights_gpu || !g_act_gpu || g_dim == 0) {
        printf("GPU not initialized or configured\n");
        return -1;
    }

    // Debug: Print first forward pass info
    if (pos == 0) {
        printf("Forward: token=%d, pos=%d, dim=%d, layers=%d\n",
               token, pos, g_dim, g_n_layers);
    }

    // Pointer shortcuts
    float* w = g_weights_gpu;
    float* a = g_act_gpu;

    int d = g_dim;
    int hd = g_hidden_dim;
    int kv = g_kv_dim;
    int hs = g_head_dim;

    // Get activation pointers
    float* x = a + g_x_offset;
    float* xb = a + g_xb_offset;
    float* xb2 = a + g_xb2_offset;
    float* q = a + g_q_offset;
    float* k = a + g_k_offset;
    float* v = a + g_v_offset;
    float* hb = a + g_hb_offset;
    float* hb2 = a + g_hb2_offset;
    float* logits = a + g_logits_offset;
    float* k_cache = a + g_k_cache_offset;
    float* v_cache = a + g_v_cache_offset;
    int* result = (int*)(a + g_result_offset);

    // Token embedding lookup (copy to x)
    float* emb = w + g_token_emb_offset + token * d;
    cudaMemcpyAsync(x, emb, d * sizeof(float), cudaMemcpyDeviceToDevice, g_stream);

    // Debug: Check first embedding values
    if (pos == 0) {
        cudaStreamSynchronize(g_stream);
        float emb_check[4];
        cudaMemcpy(emb_check, x, 4 * sizeof(float), cudaMemcpyDeviceToHost);
        printf("x[0..3] after emb lookup: %.4f, %.4f, %.4f, %.4f\n",
               emb_check[0], emb_check[1], emb_check[2], emb_check[3]);
    }

    // Forward through layers
    for (int layer = 0; layer < g_n_layers; layer++) {
        // Weight pointers for this layer
        float* rms_att = w + g_rms_att_offset + layer * d;
        float* wq = w + g_wq_offset + layer * d * d;
        float* wk = w + g_wk_offset + layer * kv * d;
        float* wv = w + g_wv_offset + layer * kv * d;
        float* wo = w + g_wo_offset + layer * d * d;
        float* rms_ffn = w + g_rms_ffn_offset + layer * d;
        float* w1 = w + g_w1_offset + layer * hd * d;
        float* w2 = w + g_w2_offset + layer * d * hd;
        float* w3 = w + g_w3_offset + layer * hd * d;

        // RMSNorm
        rmsnorm_kernel<<<1, 256, 0, g_stream>>>(xb, x, rms_att, d, 1e-6f);

        // QKV projections
        cublas_matvec(q, xb, wq, d, d);
        cublas_matvec(k, xb, wk, kv, d);
        cublas_matvec(v, xb, wv, kv, d);

        // Add biases if present
        if (g_has_bias) {
            float* bq = w + g_bq_offset + layer * d;
            float* bk = w + g_bk_offset + layer * kv;
            float* bv = w + g_bv_offset + layer * kv;
            add_bias_kernel<<<(d+255)/256, 256, 0, g_stream>>>(q, bq, d);
            add_bias_kernel<<<(kv+255)/256, 256, 0, g_stream>>>(k, bk, kv);
            add_bias_kernel<<<(kv+255)/256, 256, 0, g_stream>>>(v, bv, kv);
        }

        // RoPE
        int freq_offset = pos * (hs / 2);
        float* cos = w + g_freq_cos_offset + freq_offset;
        float* sin = w + g_freq_sin_offset + freq_offset;
        int max_heads = (g_n_heads > g_n_kv_heads) ? g_n_heads : g_n_kv_heads;
        rope_kernel<<<max_heads, hs/2, 0, g_stream>>>(q, k, cos, sin, g_n_heads, g_n_kv_heads, hs);

        // Update KV cache
        float* layer_k_cache = k_cache + layer * g_n_kv_heads * g_seq_len * hs;
        float* layer_v_cache = v_cache + layer * g_n_kv_heads * g_seq_len * hs;
        kv_cache_update_kernel<<<g_n_kv_heads, hs, 0, g_stream>>>(
            layer_k_cache, layer_v_cache, k, v, g_n_kv_heads, pos, g_seq_len, hs);

        // GQA Attention
        float scale = 1.0f / sqrtf((float)hs);
        int smem = (pos + 1 + 32) * sizeof(float);  // scores + partial sums
        gqa_attention_kernel<<<g_n_heads, 128, smem, g_stream>>>(
            xb, q, layer_k_cache, layer_v_cache,
            g_n_heads, g_n_kv_heads, pos + 1, g_seq_len, hs, scale);

        // Output projection
        cublas_matvec(xb2, xb, wo, d, d);

        // Residual add
        residual_add_kernel<<<(d+255)/256, 256, 0, g_stream>>>(x, xb2, d);

        // FFN
        rmsnorm_kernel<<<1, 256, 0, g_stream>>>(xb, x, rms_ffn, d, 1e-6f);
        cublas_matvec(hb, xb, w1, hd, d);
        cublas_matvec(hb2, xb, w3, hd, d);
        swiglu_kernel<<<(hd+255)/256, 256, 0, g_stream>>>(hb, hb, hb2, hd);
        cublas_matvec(xb, hb, w2, d, hd);
        residual_add_kernel<<<(d+255)/256, 256, 0, g_stream>>>(x, xb, d);
    }

    // Final RMSNorm
    float* rms_final = w + g_rms_final_offset;
    rmsnorm_kernel<<<1, 256, 0, g_stream>>>(xb, x, rms_final, d, 1e-6f);

    // Logits
    float* token_emb = w + g_token_emb_offset;
    cublas_matvec(logits, xb, token_emb, g_vocab_size, d);

    // Debug: Check logits before argmax
    if (pos < 3 || (pos >= 15 && pos < 20)) {
        cudaStreamSynchronize(g_stream);
        float logit_check[5];
        cudaMemcpy(logit_check, logits, 5 * sizeof(float), cudaMemcpyDeviceToHost);

        // Also check x state
        float x_check[4];
        cudaMemcpy(x_check, x, 4 * sizeof(float), cudaMemcpyDeviceToHost);

        printf("pos=%d x[0..3]: %.4f, %.4f, %.4f, %.4f  logits[0..4]: %.4f, %.4f, %.4f, %.4f, %.4f\n",
               pos, x_check[0], x_check[1], x_check[2], x_check[3],
               logit_check[0], logit_check[1], logit_check[2], logit_check[3], logit_check[4]);
    }

    // Argmax
    argmax_kernel<<<1, 256, 0, g_stream>>>(result, logits, g_vocab_size);

    // Sync and return result
    cudaStreamSynchronize(g_stream);
    int next_token;
    cudaMemcpy(&next_token, result, sizeof(int), cudaMemcpyDeviceToHost);

    return next_token;
}

}  // extern "C"
