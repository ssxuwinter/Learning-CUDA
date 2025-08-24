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
/**
 * @brief Find the k-th largest element in a vector using CUDA.
 * 
 * @tparam T Type of elements in the input vector (should support `int` and `float`).
 * @param h_input Host-side input vector.
 * @param k 1-based index of the element to find (e.g., `k=1` returns the largest element).
 * @return T The k-th largest element in `h_input`.

 * @note Must use CUDA kernels for all compute-intensive steps; no significant CPU allowed.
 * @note Library functions that can directly complete a significant part of the work are NOT allowed. 
 * @note For invalid cases, return T(-100).
 * @note Handles device memory management (allocate/copy/free) internally. Errors should be thrown.
 */
// template <typename T>
// T kthLargest(const std::vector<T>& h_input, size_t k) {
//   // TODO: Implement the kthLargest function
//   return T(-1000);
// }

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */

template <typename T>
__device__ T max_warp_reduce(T local_max){
#pragma unroll
for (int offset = warpSize / 2; offset > 0; offset /= 2){
    // printf("offset = %d\n", offset);
    T y = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
    // printf("Debug: warp reduce:offset = %d, local_max = %f, y = %f\n", offset,(float)local_max, (float)y);
    local_max = local_max > y ? local_max : y;
}
return local_max;
}
template <typename T>

__device__ T sum_warp_reduce(T local_sum){
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2){
        T y = __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
        local_sum += y;
    }
    return local_sum;
}

template <typename T, int B_r, int B_c, int D_HEAD>
__global__ void flash_attention_kernel(
    const T* __restrict__ q_ptr, 
    const T* __restrict__ k_ptr, 
    const T* __restrict__ v_ptr, 
    T* __restrict__ o_ptr,
    int target_seq_len, int src_seq_len,
    int query_heads, int kv_heads,
    bool is_causal, float scale) {

    const size_t tid = threadIdx.x;
    size_t lane_id = tid % 32;
    size_t warp_id = tid / 32;
    //每个block负责一个head的Bc行
    //退出检查
    if(blockIdx.x >= (target_seq_len + B_r - 1) / B_r) return;
    
    const int heads_per_group = query_heads / kv_heads;
    const size_t q_batch_stride = (size_t)target_seq_len * query_heads * D_HEAD;
    const size_t kv_batch_stride = (size_t)src_seq_len * kv_heads * D_HEAD;

    const int batch_id = blockIdx.z;
    const int query_head_idx = blockIdx.y;
    const int kv_head_idx = query_head_idx / heads_per_group;
    //起始ROW
    const int block_start_r = blockIdx.x * B_r;
    

    //当前block起始q,k,v指针
    q_ptr += batch_id * q_batch_stride + query_head_idx * D_HEAD;
    k_ptr += batch_id * kv_batch_stride + kv_head_idx * D_HEAD;
    v_ptr += batch_id * kv_batch_stride + kv_head_idx * D_HEAD;
    

    //共享内存
    __shared__ T q_tile[B_r][D_HEAD]; //每个block的B_r行q
    __shared__ T k_tile[B_c][D_HEAD]; //每个block的B
    __shared__ T v_tile[B_c][D_HEAD]; //每个block的B_c行v
    __shared__ T sp_tile[B_r][B_c];
    __shared__ T m_r[B_r]; //每个block的B_r行m
    __shared__ T m_old_r[B_r]; //每个block的B_r行m_old
    __shared__ T l_r[B_r]; //每个block的B_r行l
    __shared__ T output_tile[B_r][D_HEAD]; //每个block的B_r行accq
   
    //load Qi from HBM to shared memory
    for(size_t i = tid; i < B_r * D_HEAD; i += blockDim.x){
        int row = i / D_HEAD;
        int col = i % D_HEAD;
        if (block_start_r + row < target_seq_len){
            const size_t q_row_idx = (size_t)(block_start_r + row) * query_heads * D_HEAD;
            q_tile[row][col] = q_ptr[q_row_idx + col];
        } 
        else {
            q_tile[row][col] = T(0);
        }
    }
    //initialize m_r,l_r,output_tile
    for (size_t i = tid; i < B_r; i += blockDim.x){
        m_r[i] = d_lowest<T>();
        l_r[i] = T(0);
        #pragma unroll
        for (int j = 0; j < D_HEAD; j++){
            output_tile[i][j] = T(0);
        }
    }
    __syncthreads();

    const int T_r = (src_seq_len + B_c - 1) / B_c; //k,v的tile数量
    for (int j = 0; j < T_r; j++){
        //cfg
        const int block_start_c = j * B_c;
        int valid_r = min(B_r, target_seq_len - block_start_r);
        int valid_c = min(B_c, src_seq_len - block_start_c);

        //load Kj,Vj from HBM to shared memory
        for (size_t i = tid; i < B_c * D_HEAD; i += blockDim.x){
            int row = i / D_HEAD;
            int col = i % D_HEAD;
            if (j * B_c + row < src_seq_len){
                const size_t kv_row_idx = (size_t)(j * B_c + row) * kv_heads * D_HEAD;
                k_tile[row][col] = k_ptr[kv_row_idx + col];
                v_tile[row][col] = v_ptr[kv_row_idx + col];
            } 
            else {
                k_tile[row][col] = T(0);
                v_tile[row][col] = T(0);
            }
        }
        __syncthreads();

        //on chip compute S^(j) = Q_i K_j^T
        
        for (size_t idx = tid; idx < valid_r * valid_c; idx += blockDim.x){
            int row = idx / valid_c;
            int col = idx % valid_c;

            T score = T(0); 

            for (int d = 0; d < D_HEAD; d++){
                score += q_tile[row][d] * k_tile[col][d];
            }
            score *= scale;
            //causal mask
            if (is_causal && (block_start_r + row) < (block_start_c + col)){
                score = d_lowest<T>();
            }
            sp_tile[row][col] = score;
        }
        __syncthreads();
        //on chip compute m_r^(j) = max(S^(j), m_r^(j-1))
        for (size_t row = 0; row < valid_r; row++){
            if(tid == 0) m_old_r[row] = m_r[row];
            __syncthreads();
            T local_max = d_lowest<T>();
            //block stride
            for (size_t col = tid; col < valid_c; col += blockDim.x){
                local_max = local_max > sp_tile[row][col] ? local_max : sp_tile[row][col];
            }
            //warp reduce
            T warp_max = max_warp_reduce(local_max);
            //block reduce
            __shared__ T smem_max[4];
            if (lane_id == 0) smem_max[warp_id] = warp_max;
            __syncthreads();
            T block_max = d_lowest<T>();
            
            if (warp_id == 0){
                block_max = (lane_id < (blockDim.x + 31) / 32) ? smem_max[lane_id] : d_lowest<T>();
                block_max = max_warp_reduce(block_max);
                if (tid == 0) {
                    T m_new = d_lowest<T>();
                    m_new = block_max > m_old_r[row] ? block_max : m_old_r[row];
                    m_r[row] = m_new;
                }
            }
            __syncthreads();

            T local_sum = 0;
            for (size_t col = tid; col < valid_c; col += blockDim.x){
                T s_val = sp_tile[row][col];
                if (s_val > d_lowest<T>()) {
                    T p_val = expf(s_val - m_r[row]);  // 计算 P_i^(j)
                    sp_tile[row][col] = p_val;           // ✅ 原地替换存储 P_i^(j)
                    local_sum += p_val;
                } else {
                    sp_tile[row][col] = 0;               // ✅ 存储 P_i^(j) = 0
                }
            }
            //warp reduce
            T warp_sum = 0;
            warp_sum = sum_warp_reduce(local_sum);
            //block reduce
            __shared__ T smem_sum[4];
            if (lane_id == 0) smem_sum[warp_id] = warp_sum;
            __syncthreads();
            T block_sum = 0;
            if (warp_id == 0){
                block_sum = (lane_id < (blockDim.x + 31) / 32) ? smem_sum[lane_id] : 0;
                block_sum = sum_warp_reduce(block_sum);
                if (tid == 0){
                    T l_old = l_r[row];
                    T l_new = expf(m_old_r[row] - m_r[row]) * l_old + block_sum;
                    l_r[row] = l_new;
                }
            }
            __syncthreads();
        }
        for (int idx = tid; idx < valid_r * D_HEAD; idx += blockDim.x) {
            int m = idx / D_HEAD;
            int d = idx % D_HEAD;
            
            T correction_factor = expf(m_old_r[m] - m_r[m]);
            output_tile[m][d] *= correction_factor;  // 缩放旧输出
            
            // 添加新贡献
            T contribution = 0;
            for (int n = 0; n < valid_c; n++) {
                contribution += sp_tile[m][n] * v_tile[n][d];
            }
            output_tile[m][d] += contribution;
        }
        __syncthreads();
    }
    //write back to HBM
    for (size_t i = tid; i < B_r * D_HEAD; i += blockDim.x){
        int row = i / D_HEAD;
        int col = i % D_HEAD;
        if (block_start_r + row < target_seq_len){
            const size_t o_row_idx = (size_t)batch_id * q_batch_stride + (block_start_r + row) * query_heads * D_HEAD + query_head_idx * D_HEAD;
            float inv_sum = 1.0f / l_r[row];
            o_ptr[o_row_idx + col] = output_tile[row][col]*inv_sum;
        }
    }
    __syncthreads();
}

  /**
   * @brief 主机端封装函数，用于计算 Flash Attention
   */
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
      // **FIX**: 使用 switch 语句扩展对不同 head_dim 的支持
      switch (head_dim) {
       case 2: {
           constexpr int D_HEAD = 2;
           constexpr int BLOCK_SIZE_M = 128;
           constexpr int BLOCK_SIZE_N = 128;
           dim3 grid_dim((target_seq_len + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M, query_heads, batch_size);
           dim3 block_dim(128);
           const float scale = 1.0f / sqrt(static_cast<float>(D_HEAD));
           
           std::cout << "Grid: (" << grid_dim.x << ", " << grid_dim.y << ", " << grid_dim.z << ")" << std::endl;
           std::cout << "Block: (" << block_dim.x << ", " << block_dim.y << ", " << block_dim.z << ")" << std::endl;
           std::cout << "Scale: " << scale << std::endl;
           
           flash_attention_kernel<T, BLOCK_SIZE_M, BLOCK_SIZE_N, D_HEAD><<<grid_dim, block_dim>>>(
               d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, query_heads, kv_heads, is_causal, scale);
               
           // 立即检查启动错误
           cudaError_t launch_error = cudaGetLastError();
           if (launch_error != cudaSuccess) {
               // std::cout << "Kernel launch failed: " << cudaGetErrorString(launch_error) << std::endl;
               throw std::runtime_error("Kernel launch failed");
           }
           std::cout << "Kernel launched successfully" << std::endl;
           break;
       }
       case 4: {
           constexpr int D_HEAD = 4;
           constexpr int BLOCK_SIZE_M = 128;
           constexpr int BLOCK_SIZE_N = 128;
           dim3 grid_dim((target_seq_len + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M, query_heads, batch_size);
           dim3 block_dim(64);
           const float scale = 1.0f / sqrt(static_cast<float>(D_HEAD));
           std::cout << "Grid: (" << grid_dim.x << ", " << grid_dim.y << ", " << grid_dim.z << ")" << std::endl;
           std::cout << "Block: (" << block_dim.x << ", " << block_dim.y << ", " << block_dim.z << ")" << std::endl;
           std::cout << "Scale: " << scale << std::endl;
       
           flash_attention_kernel<T, BLOCK_SIZE_M, BLOCK_SIZE_N, D_HEAD><<<grid_dim, block_dim>>>(d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, query_heads, kv_heads, is_causal, scale);
           cudaError_t launch_error = cudaGetLastError();
           if (launch_error != cudaSuccess) {
               throw std::runtime_error("Kernel launch failed");
           }
           std::cout << "Kernel launched successfully" << std::endl;
           break;
       }
       case 8: {
           constexpr int D_HEAD = 8;
           constexpr int BLOCK_SIZE_M = 32;
           constexpr int BLOCK_SIZE_N = 32;
           dim3 grid_dim((target_seq_len + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M, query_heads, batch_size);
           dim3 block_dim(128);
           const float scale = 1.0f / sqrt(static_cast<float>(D_HEAD));
           flash_attention_kernel<T, BLOCK_SIZE_M, BLOCK_SIZE_N, D_HEAD><<<grid_dim, block_dim>>>(d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, query_heads, kv_heads, is_causal, scale);
           break;
       }
       case 16: {
           constexpr int D_HEAD = 16;
           constexpr int BLOCK_SIZE_M = 32;
           constexpr int BLOCK_SIZE_N = 32;
           dim3 grid_dim((target_seq_len + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M, query_heads, batch_size);
           dim3 block_dim(128);
           const float scale = 1.0f / sqrt(static_cast<float>(D_HEAD));
           flash_attention_kernel<T, BLOCK_SIZE_M, BLOCK_SIZE_N, D_HEAD><<<grid_dim, block_dim>>>(d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, query_heads, kv_heads, is_causal, scale);
           break;
       }
       case 24: {
        constexpr int D_HEAD = 24;
        constexpr int BLOCK_SIZE_M = 32;  // 适中的块大小
        constexpr int BLOCK_SIZE_N = 32;
        dim3 grid_dim((target_seq_len + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M, query_heads, batch_size);
        dim3 block_dim(128);
        const float scale = 1.0f / sqrt(static_cast<float>(D_HEAD));
        flash_attention_kernel<T, BLOCK_SIZE_M, BLOCK_SIZE_N, D_HEAD><<<grid_dim, block_dim>>>(d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, query_heads, kv_heads, is_causal, scale);
        break;
    }
       case 32: {
           constexpr int D_HEAD = 32;
           constexpr int BLOCK_SIZE_M = 32;
           constexpr int BLOCK_SIZE_N = 32;
           dim3 grid_dim((target_seq_len + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M, query_heads, batch_size);
           dim3 block_dim(128);
           const float scale = 1.0f / sqrt(static_cast<float>(D_HEAD));
           flash_attention_kernel<T, BLOCK_SIZE_M, BLOCK_SIZE_N, D_HEAD><<<grid_dim, block_dim>>>(d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, query_heads, kv_heads, is_causal, scale);
           break;
       }
       case 48: {
           constexpr int D_HEAD = 48;
           constexpr int BLOCK_SIZE_M = 32;
           constexpr int BLOCK_SIZE_N = 32;
           dim3 grid_dim((target_seq_len + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M, query_heads, batch_size);
           dim3 block_dim(128);
           const float scale = 1.0f / sqrt(static_cast<float>(D_HEAD));
           flash_attention_kernel<T, BLOCK_SIZE_M, BLOCK_SIZE_N, D_HEAD><<<grid_dim, block_dim>>>(d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, query_heads, kv_heads, is_causal, scale);
           break;
       }
       case 64: {
           constexpr int D_HEAD = 64;
           constexpr int BLOCK_SIZE_M = 32;
           constexpr int BLOCK_SIZE_N = 32;
           dim3 grid_dim((target_seq_len + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M, query_heads, batch_size);
           dim3 block_dim(128);
           const float scale = 1.0f / sqrt(static_cast<float>(D_HEAD));
           flash_attention_kernel<T, BLOCK_SIZE_M, BLOCK_SIZE_N, D_HEAD><<<grid_dim, block_dim>>>(d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, query_heads, kv_heads, is_causal, scale);
           break;
       }
       case 80: {
           constexpr int D_HEAD = 80;
           constexpr int BLOCK_SIZE_M = 16;
           constexpr int BLOCK_SIZE_N = 16;
           dim3 grid_dim((target_seq_len + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M, query_heads, batch_size);
           dim3 block_dim(128);
           const float scale = 1.0f / sqrt(static_cast<float>(D_HEAD));
           flash_attention_kernel<T, BLOCK_SIZE_M, BLOCK_SIZE_N, D_HEAD><<<grid_dim, block_dim>>>(d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, query_heads, kv_heads, is_causal, scale);
           break;
       }
       case 128: {
           constexpr int D_HEAD = 128;
           constexpr int BLOCK_SIZE_M = 16;
           constexpr int BLOCK_SIZE_N = 16;
           dim3 grid_dim((target_seq_len + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M, query_heads, batch_size);
           dim3 block_dim(64);
           const float scale = 1.0f / sqrt(static_cast<float>(D_HEAD));
           flash_attention_kernel<T, BLOCK_SIZE_M, BLOCK_SIZE_N, D_HEAD><<<grid_dim, block_dim>>>(d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, query_heads, kv_heads, is_causal, scale);
           break;
       }
       default:
           throw std::invalid_argument("Unsupported head_dim: " + std::to_string(head_dim) + 
                                     ". Supported values are: 2, 4, 8, 16, 32, 48, 64, 80, 96, 128, 160, 256");
   }
      std::cout << "Kernel launched successfully1" << std::endl;
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      std::cout<< "hello?" << std::endl;
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
// template int kthLargest<int>(const std::vector<int>&, size_t);
// template float kthLargest<float>(const std::vector<float>&, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
