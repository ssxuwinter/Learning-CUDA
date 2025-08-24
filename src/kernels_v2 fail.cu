#include <vector>
#include <cuda_runtime.h>
#include <stdexcept>
#include <algorithm>
#include <cfloat>
#include <iostream>
#include <cuda_fp16.h>
#include <limits.h> 
#include "../tester/utils.h"

#include <cmath>
#include <numeric>
#include <iomanip>

template <typename T>
__device__ __forceinline__ T d_lowest();

template <>
__device__ __forceinline__ int d_lowest<int>() { return INT_MIN; }

template <>
__device__ __forceinline__ float d_lowest<float>() { return -FLT_MAX; }

template <typename T>
T h_lowest();

template <>
int h_lowest<int>() { return INT_MIN; }

template <>
float h_lowest<float>() { return -FLT_MAX; }

template <typename T>
__device__ T max_warp_reduce(T local_max){
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2){
        T y = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
        local_max = local_max > y ? local_max : y;
    }
    // 广播给所有lanes
    return __shfl_sync(0xFFFFFFFF, local_max, 0);
}

template <typename T>
__device__ T sum_warp_reduce(T local_sum){
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2){
        T y = __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
        local_sum += y;
    }
    // 广播给所有lanes
    return __shfl_sync(0xFFFFFFFF, local_sum, 0);
}

template <typename T, int B_r, int B_c, int D_HEAD, int ELEM_PER_THREAD >
__global__ void flash_attention_kernel(
    const T* __restrict__ q_ptr, 
    const T* __restrict__ k_ptr, 
    const T* __restrict__ v_ptr, 
    T* __restrict__ o_ptr,
    int target_seq_len, int src_seq_len,
    int query_heads, int kv_heads,
    bool is_causal, float scale) {

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = blockDim.x / 32;
    
    // 每个warp处理Q的特定行范围
    const int rows_per_warp = B_r / num_warps;
    const int warp_row_start = warp_id * rows_per_warp;
    
    // 解码grid
    const int block_start_r = blockIdx.x * B_r;
    const int batch_head_idx = blockIdx.z;
    const int batch_id = batch_head_idx / query_heads;
    const int query_head_idx = batch_head_idx % query_heads;
    const int heads_per_group = query_heads / kv_heads;
    const int kv_head_idx = query_head_idx / heads_per_group;

    // 指针偏移
    const size_t q_batch_stride = (size_t)target_seq_len * query_heads * D_HEAD;
    const size_t kv_batch_stride = (size_t)src_seq_len * kv_heads * D_HEAD;
    q_ptr += batch_id * q_batch_stride + query_head_idx * D_HEAD;
    k_ptr += batch_id * kv_batch_stride + kv_head_idx * D_HEAD;
    v_ptr += batch_id * kv_batch_stride + kv_head_idx * D_HEAD;

    // 共享内存 - 所有数据都存储在共享内存中
    __shared__ T q_tile[B_r][D_HEAD];
    __shared__ T k_tile[B_c][D_HEAD];
    __shared__ T v_tile[B_c][D_HEAD];
    
    
    //寄存器内存
    T R_SP [ELEM_PER_THREAD]; // 每个线程存储的输出行数
    T m_old = d_lowest<T>();
    T m_new = d_lowest<T>();
    T l_old = T(0);
    T l_new = T(0);

    int valid_r = min(B_r, target_seq_len - block_start_r);

    // 协作加载Q tile - 所有线程参与
    for (int idx = tid; idx < B_r * D_HEAD; idx += blockDim.x) {
        int row = idx / D_HEAD;
        int col = idx % D_HEAD;
        if (block_start_r + row < target_seq_len) {
            const size_t q_offset = ((size_t)(block_start_r + row) * query_heads * D_HEAD);
            q_tile[row][col] = q_ptr[q_offset + col];
        } 
        else {
            q_tile[row][col] = T(0);
        }
    }
    

    // Flash Attention主循环 - 遍历K的块
    const int T_c = (src_seq_len + B_c - 1) / B_c;
    for (int j = 0; j < T_c; j++) {
        const int block_start_c = j * B_c;
        int valid_c = min(B_c, src_seq_len - block_start_c);

        // 协作加载KV tiles - 所有线程参与
        for (int idx = tid; idx < B_c * D_HEAD; idx += blockDim.x) {
            int row = idx / D_HEAD;
            int col = idx % D_HEAD;
            if (block_start_c + row < src_seq_len) {
                const size_t kv_offset = ((size_t)(block_start_c + row) * kv_heads * D_HEAD);
                k_tile[row][col] = k_ptr[kv_offset + col];
                v_tile[row][col] = v_ptr[kv_offset + col];
            } else {
                k_tile[row][col] = T(0);
                v_tile[row][col] = T(0);
            }
        }
        __syncthreads();
        // 每个线程计算自己的ELEM_PER_THREAD
        for (int i = 0; i < ELEM_PER_THREAD; i++) {
            int row_warp = lane_id / 2;
            int col_warp = (lane_id % 2)*ELEM_PER_THREAD + i;
            int row = warp_row_start + row_warp;
            int col = col_warp;
            if (row < valid_r) {
                T score = T(0);
                for (int d = 0; d < D_head; d++){
                    score += q_tile[row][d] * k_tile[col][d];
                }
                score *= scale;
                if (is_causal && (block_start_r + row) < (block_start_c + col)) {
                    score = d_lowest<T>();
                }
                R_SP[i] = score;
                m_new = m_new > score ? m_new : score;
            }
            else{
                R_SPP[i] = T(0);
                continue;
            }
        }
        // 计算m_new的warp内两个线程进行规约
        int partner_lane = (lane_id & ~1) | (1 - (lane_id & 1));  // 翻转最低位，配对相邻线程
        T partner_max = __shfl_sync(0xFFFFFFFF, m_new, partner_lane);
        m_new = m_new > partner_max ? m_new : partner_max;

        int row = warp_row_start + lane_id/2;
        if (lane_id % 2 == 0) {            
            m_old = m_shared[row];
            l_old = l_shared[row];
            
            // 更新全局最大值
            m_new = max(m_old, m_new);
            m_shared[row] = m_new;
        }
        m_old = __shfl_sync(0xFFFFFFFF, m_old, (lane_id & ~1)); // 从偶数线程广播
        l_old = __shfl_sync(0xFFFFFFFF, l_old, (lane_id & ~1)); // 从偶数线程广播
        
        T l_sum = T(0);
        for (int i = 0; i < ELEM_PER_THREAD; i++) {
            int col_warp = (lane_id % 2)*ELEM_PER_THREAD + i;
            int col = col_warp;
            if (col < valid_c) {
                T s_val = R_SP[i];
                if(s_val == d_lowest<T>()){
                    R_SP[i] = T(0);
                }
                else{
                    s_val = expf(s_val - m_new);
                    R_SP[i] = s_val;
                    l_sum += s_val;
                }
            }
        }
        T partner_sum = __shfl_sync(0xFFFFFFFF, l_sum, partner_lane);
        l_sum += partner_sum;

        T correction = T(1);
        if (m_old > d_lowest<T>()) {
            correction = expf(m_old - m_new);
        }
        
        // 更新归一化因子
        l_new = correction * l_old + l_sum;
        if (lane_id % 2 == 0) {
            l_shared[row] = l_new;
        }
        l_new = __shfl_sync(0xFFFFFFFF, l_new, (lane_id & ~1));

        //计算输出
        for (int i = 0; i < ELEM_PER_THREAD; i++) {
            int row_warp = lane_id / 2;
            int col_warp = (lane_id % 2)*ELEM_PER_THREAD + i;
            int row = warp_row_start + row_warp;
            int col = col_warp;
            T correction_factor = expf(m_old - m_new);
            if (j > 0){
                output_tile[row][col] *= correction_factor;
            }
            T contribution = 0;
            for (int n = 0; n < valid_c; n++) {
                contribution += sp_tile[m][n] * v_tile[n][d];
            }
            output_tile[m][d] += contribution;
        }





       
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {

    static int call_count = 0;
    call_count++;
    
    std::cout << "=== flashAttention call #" << call_count << " ===" << std::endl;
    std::cout << "batch_size: " << batch_size << ", target_seq_len: " << target_seq_len 
              << ", src_seq_len: " << src_seq_len << ", query_heads: " << query_heads 
              << ", kv_heads: " << kv_heads << ", head_dim: " << head_dim 
              << ", is_causal: " << std::boolalpha << is_causal << std::endl;
    std::cout<< "computing tensor sizes..." << std::endl;
    
    size_t q_size = (size_t)batch_size * target_seq_len * query_heads * head_dim;
    size_t k_size = (size_t)batch_size * src_seq_len * kv_heads * head_dim;
    size_t v_size = (size_t)batch_size * src_seq_len * kv_heads * head_dim;
    size_t o_size = (size_t)batch_size * target_seq_len * query_heads * head_dim;
    
    std::cout << "q_size: " << q_size << ", k_size: " << k_size 
              << ", v_size: " << v_size << ", o_size: " << o_size << std::endl;
    
    if (h_q.size() != q_size || h_k.size() != k_size || h_v.size() != v_size) {
        throw std::invalid_argument("Input tensor dimensions do not match vector sizes.");
    }
    h_o.resize(o_size);
    std::cout<< "Allocated output tensor of size: " << h_o.size() << std::endl;
    
    T *d_q, *d_k, *d_v, *d_o;
    CUDA_CHECK(cudaMalloc(&d_q, q_size * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_k, k_size * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_v, v_size * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_o, o_size * sizeof(T)));
    
    std::cout<< "Copying data to device..." << std::endl;
    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), q_size * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), k_size * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), v_size * sizeof(T), cudaMemcpyHostToDevice));
    
    std::cout<<"Launching kernel..." << std::endl;
    
    // 使用共享内存版本的split-Q策略
    int total_batch_heads = batch_size * query_heads;
    const float scale = 1.0f / sqrt(static_cast<float>(head_dim));
    
    switch (head_dim) {
    case 2: {
        constexpr int D_HEAD = 2;
        constexpr int BLOCK_SIZE_M = 32;  // 减小块大小以适应共享内存
        constexpr int BLOCK_SIZE_N = 64;
        dim3 grid_dim(
            (target_seq_len + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M,
            1,
            total_batch_heads
        );
        dim3 block_dim(64);  // 2个warp
        
        flash_attention_kernel<T, BLOCK_SIZE_M, BLOCK_SIZE_N, D_HEAD><<<grid_dim, block_dim>>>(
            d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, query_heads, kv_heads, is_causal, scale);
        break;
    }
    case 4: {
        constexpr int D_HEAD = 4;
        constexpr int BLOCK_SIZE_M = 32;
        constexpr int BLOCK_SIZE_N = 64;
        dim3 grid_dim(
            (target_seq_len + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M,
            1,
            total_batch_heads
        );
        dim3 block_dim(64);
        
        flash_attention_kernel<T, BLOCK_SIZE_M, BLOCK_SIZE_N, D_HEAD><<<grid_dim, block_dim>>>(
            d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, query_heads, kv_heads, is_causal, scale);
        break;
    }
    case 8: {
        constexpr int D_HEAD = 8;
        constexpr int BLOCK_SIZE_M = 32;
        constexpr int BLOCK_SIZE_N = 64;
        dim3 grid_dim(
            (target_seq_len + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M,
            1,
            total_batch_heads
        );
        dim3 block_dim(64);
        
        flash_attention_kernel<T, BLOCK_SIZE_M, BLOCK_SIZE_N, D_HEAD><<<grid_dim, block_dim>>>(
            d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, query_heads, kv_heads, is_causal, scale);
        break;
    }
    case 16: {
        constexpr int D_HEAD = 16;
        constexpr int BLOCK_SIZE_M = 32;
        constexpr int BLOCK_SIZE_N = 64;
        dim3 grid_dim(
            (target_seq_len + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M,
            1,
            total_batch_heads
        );
        dim3 block_dim(64);
        
        flash_attention_kernel<T, BLOCK_SIZE_M, BLOCK_SIZE_N, D_HEAD><<<grid_dim, block_dim>>>(
            d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, query_heads, kv_heads, is_causal, scale);
        break;
    }
    case 32: {
        constexpr int D_HEAD = 32;
        constexpr int BLOCK_SIZE_M = 32;
        constexpr int BLOCK_SIZE_N = 32;  // 减小以适应共享内存
        dim3 grid_dim(
            (target_seq_len + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M,
            1,
            total_batch_heads
        );
        dim3 block_dim(64);  // 2个warp
        
        flash_attention_kernel<T, BLOCK_SIZE_M, BLOCK_SIZE_N, D_HEAD><<<grid_dim, block_dim>>>(
            d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, query_heads, kv_heads, is_causal, scale);
        break;
    }
    case 64: {
        constexpr int D_HEAD = 64;
        constexpr int BLOCK_SIZE_M = 16;  // 进一步减小
        constexpr int BLOCK_SIZE_N = 32;
        dim3 grid_dim(
            (target_seq_len + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M,
            1,
            total_batch_heads
        );
        dim3 block_dim(64);
        
        flash_attention_kernel<T, BLOCK_SIZE_M, BLOCK_SIZE_N, D_HEAD><<<grid_dim, block_dim>>>(
            d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, query_heads, kv_heads, is_causal, scale);
        break;
    }
    case 128: {
        constexpr int D_HEAD = 128;
        constexpr int BLOCK_SIZE_M = 16;
        constexpr int BLOCK_SIZE_N = 16;  // 最小块大小
        dim3 grid_dim(
            (target_seq_len + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M,
            1,
            total_batch_heads
        );
        dim3 block_dim(32);  // 1个warp
        
        flash_attention_kernel<T, BLOCK_SIZE_M, BLOCK_SIZE_N, D_HEAD><<<grid_dim, block_dim>>>(
            d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, query_heads, kv_heads, is_causal, scale);
        break;
    }
    default:
        throw std::invalid_argument("Unsupported head_dim: " + std::to_string(head_dim));
    }
    
    std::cout << "Kernel launched successfully" << std::endl;
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout<< "Kernel execution completed" << std::endl;
    CUDA_CHECK(cudaMemcpy(h_o.data(), d_o, o_size * sizeof(T), cudaMemcpyDeviceToHost));
    std::cout << "Copying output data back to host..." << std::endl;
    
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_o));
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);