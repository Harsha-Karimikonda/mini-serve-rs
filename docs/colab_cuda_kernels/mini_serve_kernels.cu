// Standalone Custom CUDA Kernels for Mini-Serve (RLSE)
// 1. fused_rmsnorm_kernel: Replaces 6-op decomposition in src/backends/qwen2.rs::RmsNorm::forward
// 2. paged_gqa_decode_attention_kernel: Connects src/cache/kv_cache.rs (block_size=16)
//    directly to Qwen3 GQA decode attention (src/backends/qwen2.rs::decode_attention_gqa_zero_copy)

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

__device__ __forceinline__ float to_f32(float v) { return v; }
__device__ __forceinline__ float to_f32(__half v) { return __half2float(v); }

template<typename T>
__device__ __forceinline__ T from_f32(float v);
template<>
__device__ __forceinline__ float from_f32<float>(float v) { return v; }
template<>
__device__ __forceinline__ __half from_f32<__half>(float v) { return __float2half(v); }

// 1. Fused RMSNorm CUDA Kernel (replaces 6-op decomposition in src/backends/qwen2.rs::RmsNorm)
__global__ void fused_rmsnorm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int hidden_size,
    float eps
) {
    extern __shared__ float s_sum[];
    int tid = threadIdx.x;
    int row = blockIdx.x;
    const float* x_row = x + row * hidden_size;
    float* out_row = out + row * hidden_size;

    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float v = x_row[i];
        sum_sq += v * v;
    }
    s_sum[tid] = sum_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_sum[tid] += s_sum[tid + stride];
        __syncthreads();
    }

    float inv_rms = rsqrtf(s_sum[0] / (float)hidden_size + eps);
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        out_row[i] = x_row[i] * inv_rms * weight[i];
    }
}

// 2. Non-Contiguous Paged GQA Decode Attention Kernel (BLOCK_SIZE = 16)
// Connects KVCache logical block_table (src/cache/kv_cache.rs) to Qwen3 GQA (src/backends/qwen2.rs)
// Grid: (num_heads), Block: (128 threads = head_dim)
template<int BLOCK_SIZE, int HEAD_DIM>
__global__ void paged_gqa_decode_attention_kernel(
    const float* __restrict__ q,          // [num_heads, HEAD_DIM]
    const float* __restrict__ k_pool,     // [max_blocks, num_kv_heads, BLOCK_SIZE, HEAD_DIM]
    const float* __restrict__ v_pool,     // [max_blocks, num_kv_heads, BLOCK_SIZE, HEAD_DIM]
    const int* __restrict__ block_table,  // [num_logical_blocks] -> physical block IDs
    float* __restrict__ out,              // [num_heads, HEAD_DIM]
    int num_kv_heads,
    int num_kv_groups,
    int seq_len,
    float scale
) {
    int head_idx = blockIdx.x;
    int kv_head_idx = head_idx / num_kv_groups;
    int d = threadIdx.x; // 0 .. HEAD_DIM-1

    extern __shared__ float smem[];
    float* s_reduce = smem;               // [HEAD_DIM]
    float* s_logits = smem + HEAD_DIM;    // [seq_len]

    float q_val = q[head_idx * HEAD_DIM + d];

    // Step 1: Compute Q * K^T across non-contiguous physical pages via block_table
    for (int token_pos = 0; token_pos < seq_len; ++token_pos) {
        int logical_block = token_pos / BLOCK_SIZE;
        int block_offset  = token_pos % BLOCK_SIZE;
        int physical_block = block_table[logical_block];

        int kv_idx = ((physical_block * num_kv_heads + kv_head_idx) * BLOCK_SIZE + block_offset) * HEAD_DIM + d;
        s_reduce[d] = q_val * k_pool[kv_idx];
        __syncthreads();

        for (int stride = HEAD_DIM / 2; stride > 0; stride >>= 1) {
            if (d < stride) s_reduce[d] += s_reduce[d + stride];
            __syncthreads();
        }
        if (d == 0) {
            s_logits[token_pos] = s_reduce[0] * scale;
        }
        __syncthreads();
    }

    // Step 2: Numerically stable Softmax over s_logits[0..seq_len]
    if (d == 0) {
        float max_val = -1e30f;
        for (int i = 0; i < seq_len; ++i) {
            if (s_logits[i] > max_val) max_val = s_logits[i];
        }
        float sum_exp = 0.0f;
        for (int i = 0; i < seq_len; ++i) {
            float e = expf(s_logits[i] - max_val);
            s_logits[i] = e;
            sum_exp += e;
        }
        for (int i = 0; i < seq_len; ++i) {
            s_logits[i] /= sum_exp;
        }
    }
    __syncthreads();

    // Step 3: Weighted sum over non-contiguous V pages
    float acc = 0.0f;
    for (int token_pos = 0; token_pos < seq_len; ++token_pos) {
        int logical_block = token_pos / BLOCK_SIZE;
        int block_offset  = token_pos % BLOCK_SIZE;
        int physical_block = block_table[logical_block];

        int kv_idx = ((physical_block * num_kv_heads + kv_head_idx) * BLOCK_SIZE + block_offset) * HEAD_DIM + d;
        acc += s_logits[token_pos] * v_pool[kv_idx];
    }
    out[head_idx * HEAD_DIM + d] = acc;
}

extern "C" void launch_mini_serve_kernels(
    const float* d_x, const float* d_w, float* d_rms_out, int hidden_size, float eps,
    const float* d_q, const float* d_k_pool, const float* d_v_pool, const int* d_block_table,
    float* d_attn_out, int num_heads, int num_kv_heads, int seq_len, float scale
) {
    fused_rmsnorm_kernel<<<1, 256, 256 * sizeof(float)>>>(d_x, d_w, d_rms_out, hidden_size, eps);
    int num_kv_groups = num_heads / num_kv_heads;
    size_t smem_bytes = (128 + seq_len) * sizeof(float);
    paged_gqa_decode_attention_kernel<16, 128><<<num_heads, 128, smem_bytes>>>(
        d_q, d_k_pool, d_v_pool, d_block_table, d_attn_out,
        num_kv_heads, num_kv_groups, seq_len, scale
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

// 3. F16 Fused RMSNorm with F32 variance accumulation and reference casting
__global__ void fused_rmsnorm_f16_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ weight,
    __half* __restrict__ out,
    int hidden_size,
    float eps
) {
    extern __shared__ float s_sum[];
    int tid = threadIdx.x;
    int row = blockIdx.x;
    const __half* x_row = x + row * hidden_size;
    __half* out_row = out + row * hidden_size;

    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float v = __half2float(x_row[i]);
        sum_sq += v * v;
    }
    s_sum[tid] = sum_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_sum[tid] += s_sum[tid + stride];
        __syncthreads();
    }

    float inv_rms = rsqrtf(s_sum[0] / (float)hidden_size + eps);
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        __half x_normed = __float2half(__half2float(x_row[i]) * inv_rms);
        out_row[i] = __float2half(__half2float(x_normed) * __half2float(weight[i]));
    }
}

// 4. Ragged Batched Tiled Online-Softmax Paged GQA Decode Attention
// Constant shared memory (HEAD_DIM * sizeof(float) = 512 bytes) independent of seq_len.
// Grid: (num_heads, batch_size), Block: (HEAD_DIM)
template<int BLOCK_SIZE, int HEAD_DIM, typename Scalar>
__global__ void paged_gqa_decode_tiled_kernel(
    const Scalar* __restrict__ q,          // [batch_size, num_heads, HEAD_DIM]
    const Scalar* __restrict__ k_pool,     // [max_blocks, num_kv_heads, BLOCK_SIZE, HEAD_DIM]
    const Scalar* __restrict__ v_pool,     // [max_blocks, num_kv_heads, BLOCK_SIZE, HEAD_DIM]
    const int* __restrict__ block_tables,  // [batch_size, max_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [batch_size]
    Scalar* __restrict__ out,              // [batch_size, num_heads, HEAD_DIM]
    int num_heads,
    int num_kv_heads,
    int num_kv_groups,
    int max_blocks_per_seq,
    float scale
) {
    int head_idx = blockIdx.x;
    int batch_idx = blockIdx.y;
    int kv_head_idx = head_idx / num_kv_groups;
    int d = threadIdx.x;

    extern __shared__ float s_reduce[];
    int seq_len = seq_lens[batch_idx];
    const int* block_table = block_tables + batch_idx * max_blocks_per_seq;
    float q_val = to_f32(q[(batch_idx * num_heads + head_idx) * HEAD_DIM + d]);

    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    float acc = 0.0f;

    for (int token_pos = 0; token_pos < seq_len; ++token_pos) {
        int logical_block = token_pos / BLOCK_SIZE;
        int block_offset  = token_pos % BLOCK_SIZE;
        int physical_block = block_table[logical_block];
        int kv_idx = ((physical_block * num_kv_heads + kv_head_idx) * BLOCK_SIZE + block_offset) * HEAD_DIM + d;

        s_reduce[d] = q_val * to_f32(k_pool[kv_idx]);
        __syncthreads();

        for (int stride = HEAD_DIM / 2; stride > 0; stride >>= 1) {
            if (d < stride) s_reduce[d] += s_reduce[d + stride];
            __syncthreads();
        }

        float score = s_reduce[0] * scale;
        float m_new = fmaxf(m_prev, score);
        float alpha = (m_prev == -INFINITY) ? 0.0f : expf(m_prev - m_new);
        float beta  = expf(score - m_new);
        l_prev = l_prev * alpha + beta;
        acc = acc * alpha + beta * to_f32(v_pool[kv_idx]);
        m_prev = m_new;
        __syncthreads();
    }

    out[(batch_idx * num_heads + head_idx) * HEAD_DIM + d] =
        (l_prev > 0.0f) ? from_f32<Scalar>(acc / l_prev) : from_f32<Scalar>(0.0f);
}

extern "C" int launch_rmsnorm_f32(
    const float* d_x, const float* d_w, float* d_out,
    int num_rows, int hidden_size, float eps
) {
    if (num_rows <= 0 || hidden_size <= 0) return (int)cudaErrorInvalidValue;
    fused_rmsnorm_kernel<<<num_rows, 256, 256 * sizeof(float)>>>(d_x, d_w, d_out, hidden_size, eps);
    return (int)cudaGetLastError();
}

extern "C" int launch_rmsnorm_f16(
    const void* d_x, const void* d_w, void* d_out,
    int num_rows, int hidden_size, float eps
) {
    if (num_rows <= 0 || hidden_size <= 0) return (int)cudaErrorInvalidValue;
    fused_rmsnorm_f16_kernel<<<num_rows, 256, 256 * sizeof(float)>>>(
        (const __half*)d_x, (const __half*)d_w, (__half*)d_out, hidden_size, eps
    );
    return (int)cudaGetLastError();
}

extern "C" int launch_paged_gqa_decode_tiled_f32(
    const float* d_q, const float* d_k_pool, const float* d_v_pool,
    const int* d_block_tables, const int* d_seq_lens, float* d_out,
    int batch_size, int num_heads, int num_kv_heads, int max_blocks_per_seq, float scale
) {
    if (batch_size <= 0 || num_heads <= 0 || num_kv_heads <= 0 || num_heads % num_kv_heads != 0)
        return (int)cudaErrorInvalidValue;
    dim3 grid(num_heads, batch_size);
    paged_gqa_decode_tiled_kernel<16, 128, float><<<grid, 128, 128 * sizeof(float)>>>(
        d_q, d_k_pool, d_v_pool, d_block_tables, d_seq_lens, d_out,
        num_heads, num_kv_heads, num_heads / num_kv_heads, max_blocks_per_seq, scale
    );
    return (int)cudaGetLastError();
}

extern "C" int launch_paged_gqa_decode_tiled_f16(
    const void* d_q, const void* d_k_pool, const void* d_v_pool,
    const int* d_block_tables, const int* d_seq_lens, void* d_out,
    int batch_size, int num_heads, int num_kv_heads, int max_blocks_per_seq, float scale
) {
    if (batch_size <= 0 || num_heads <= 0 || num_kv_heads <= 0 || num_heads % num_kv_heads != 0)
        return (int)cudaErrorInvalidValue;
    dim3 grid(num_heads, batch_size);
    paged_gqa_decode_tiled_kernel<16, 128, __half><<<grid, 128, 128 * sizeof(float)>>>(
        (const __half*)d_q, (const __half*)d_k_pool, (const __half*)d_v_pool,
        d_block_tables, d_seq_lens, (__half*)d_out,
        num_heads, num_kv_heads, num_heads / num_kv_heads, max_blocks_per_seq, scale
    );
    return (int)cudaGetLastError();
}

