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

 template <typename T, int BLOCK_SIZE_M, int BLOCK_SIZE_N, int D_HEAD>
 __global__ void flash_attention_kernel(
     const T* __restrict__ q_ptr, 
     const T* __restrict__ k_ptr, 
     const T* __restrict__ v_ptr, 
     T* __restrict__ o_ptr,
     int target_seq_len, int src_seq_len,
     int query_heads, int kv_heads,
     bool is_causal, float scale) {
 
     // 添加早期退出检查，避免部分线程提前退出导致死锁
     if (blockIdx.x >= (target_seq_len + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M) {
         return;
     }
     
     const int batch_idx = blockIdx.z;
     const int query_head_idx = blockIdx.y;
     const int heads_per_group = query_heads / kv_heads;
     const int kv_head_idx = query_head_idx / heads_per_group;
     const int block_start_m = blockIdx.x * BLOCK_SIZE_M;
 
     const size_t q_batch_head_stride = (size_t)target_seq_len * query_heads * D_HEAD;
     const size_t kv_batch_head_stride = (size_t)src_seq_len * kv_heads * D_HEAD;
 
     q_ptr += batch_idx * q_batch_head_stride + query_head_idx * D_HEAD;
     k_ptr += batch_idx * kv_batch_head_stride + kv_head_idx * D_HEAD;
     v_ptr += batch_idx * kv_batch_head_stride + kv_head_idx * D_HEAD;
 
     __shared__ T q_tile[BLOCK_SIZE_M][D_HEAD];
     __shared__ T k_tile[BLOCK_SIZE_N][D_HEAD];
     __shared__ T v_tile[BLOCK_SIZE_N][D_HEAD];
     __shared__ float row_max[BLOCK_SIZE_M];
     __shared__ float row_sum[BLOCK_SIZE_M];
     __shared__ T output_buffer[BLOCK_SIZE_M][D_HEAD];
 
     const int tid = threadIdx.x;
 
     // 初始化（确保所有线程都参与）
     for (int m = tid; m < BLOCK_SIZE_M; m += blockDim.x) {
         row_max[m] = d_lowest<float>();
         row_sum[m] = 0.0f;
         for (int d = 0; d < D_HEAD; d++) {
             output_buffer[m][d] = 0.0f;
         }
     }
 
     // 加载Q块
     //一个 block 内的所有线程一起，把一个 [BLOCK_SIZE_M × D_HEAD] 的 Q tile 从 global memory 搬到 shared memory
     //从block_start_m这个block里
     //block stride
     for (int row = tid; row < BLOCK_SIZE_M * D_HEAD; row += blockDim.x) {
         int m = row / D_HEAD;
         int d = row % D_HEAD;
         if (block_start_m + m < target_seq_len) {
             const size_t q_offset = ((size_t)(block_start_m + m) * query_heads * D_HEAD);
             q_tile[m][d] = q_ptr[q_offset + d];
         } else {
             q_tile[m][d] = 0.0f;
         }
     }
     __syncthreads();  // 确保所有线程都能到达这里
     
     const int num_blocks_n = (src_seq_len + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N; 
     for (int j = 0; j < num_blocks_n; ++j) {
         const int block_start_n = j * BLOCK_SIZE_N;
 
         // 加载K,V块
         for (int row = tid; row < BLOCK_SIZE_N * D_HEAD; row += blockDim.x) {
             int n = row / D_HEAD;
             int d = row % D_HEAD;
             if (block_start_n + n < src_seq_len) {
                 const size_t kv_offset = ((size_t)(block_start_n + n) * kv_heads * D_HEAD);
                 k_tile[n][d] = k_ptr[kv_offset + d];
                 v_tile[n][d] = v_ptr[kv_offset + d];
             } else {
                 k_tile[n][d] = 0.0f;
                 v_tile[n][d] = 0.0f;
             }
         }
         __syncthreads();
 
         // 处理每一行 - 实现在线 softmax 更新
            // 处理每一行 - 修复版本
        for (int m = 0; m < min(BLOCK_SIZE_M, target_seq_len - block_start_m); m++) {
            __shared__ float scores[BLOCK_SIZE_N];
            __shared__ float shared_new_max;  // ✅ 共享变量，所有线程可访问
            __shared__ float shared_local_sum; // ✅ 共享变量存储 reduce 结果
            
            float local_max = d_lowest<float>();
            int valid_n = min(BLOCK_SIZE_N, src_seq_len - block_start_n);
            
            // 步骤1：计算 QK^T scores 和找最大值
            for(int n = tid; n < valid_n; n += blockDim.x) {  // ✅ 修复循环
                float score = 0.0f;
                for (int d = 0; d < D_HEAD; d++) {
                    score += static_cast<float>(q_tile[m][d]) * static_cast<float>(k_tile[n][d]);
                }
                score *= scale;
                
                if (is_causal && (block_start_m + m) < (block_start_n + n)) {
                    score = d_lowest<float>();
                }
                scores[n] = score;
                if (score > d_lowest<float>()) {
                    local_max = fmaxf(local_max, score);
                }
            }
            
            // 步骤2：Warp reduce 最大值
            for(int offset = 16; offset > 0; offset /= 2) {  // ✅ 修复 offset
                local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
            }
            
            // 步骤3：Block reduce 最大值
            __shared__ float warp_maxes[8];
            int warp_id = tid / 32;
            int lane_id = tid % 32;
            
            if (lane_id == 0) {
                warp_maxes[warp_id] = local_max;
            }
            __syncthreads();
            
            float block_max = d_lowest<float>();
            if (warp_id == 0) {  // ✅ 让第一个 warp 处理
                int num_warps = (blockDim.x + 31) / 32;
                float warp_max = (lane_id < num_warps) ? warp_maxes[lane_id] : d_lowest<float>();
                
                for(int offset = 16; offset > 0; offset /= 2) {
                    warp_max = fmaxf(warp_max, __shfl_down_sync(0xFFFFFFFF, warp_max, offset));
                }
                
                if (lane_id == 0) {
                    block_max = warp_max;
                }
            }
            __syncthreads();
            
            // 步骤4：更新全局最大值
            if(tid == 0) {
                float old_max = row_max[m];
                shared_new_max = fmaxf(old_max, block_max);  // ✅ 使用共享变量
                row_max[m] = shared_new_max;
            }
            __syncthreads();
            
            // 步骤5：计算 softmax（所有线程参与）
            float local_sum = 0.0f;
            for (int n = tid; n < valid_n; n += blockDim.x) {  // ✅ 修复循环
                if (scores[n] > d_lowest<float>()) {
                    scores[n] = expf(scores[n] - shared_new_max);  // ✅ 使用共享变量
                    local_sum += scores[n];
                } else {
                    scores[n] = 0.0f;
                }
            }
            
            // 步骤6：Warp reduce sum
            for(int offset = 16; offset > 0; offset /= 2) {
                local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
            }
            
            // 步骤7：Block reduce sum
            __shared__ float warp_sums[8];
            if (lane_id == 0) {
                warp_sums[warp_id] = local_sum;
            }
            __syncthreads();
            
            if (warp_id == 0) {
                int num_warps = (blockDim.x + 31) / 32;
                float warp_sum = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
                
                for(int offset = 16; offset > 0; offset /= 2) {
                    warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, offset);  // ✅ 修复变量名
                }
                
                if (lane_id == 0) {
                    shared_local_sum = warp_sum;  // ✅ 存储到共享变量
                }
            }
            __syncthreads();
            
            // 步骤8：更新全局状态和输出
            if(tid == 0) {
                float old_max = row_max[m];  // 注意：这里已经是更新后的值
                float old_sum = row_sum[m];
                float correction_factor = expf(old_max - shared_new_max);
                float final_new_sum = old_sum * correction_factor + shared_local_sum;  // ✅ 使用不同变量名
                
                row_sum[m] = final_new_sum;
                
                // 更新输出
                if (final_new_sum > 0.0f) {
                    float output_scale = correction_factor;
                    
                    for (int d = 0; d < D_HEAD; d++) {
                        output_buffer[m][d] *= output_scale;
                        
                        float new_contribution = 0.0f;
                        for (int n = 0; n < valid_n; n++) {
                            new_contribution += scores[n] * static_cast<float>(v_tile[n][d]);
                        }
                        output_buffer[m][d] += new_contribution;
                    }
                }
            }
            __syncthreads();
        }

    }
    for (int m = tid; m < min(BLOCK_SIZE_M, target_seq_len - block_start_m); m += blockDim.x) {
        if (row_sum[m] > 0.0f) {
            float inv_sum = 1.0f / row_sum[m];
            for (int d = 0; d < D_HEAD; d++) {
                output_buffer[m][d] *= inv_sum;
            }
        }
    }
    __syncthreads();
 
     // 写回结果
     for (int row = tid; row < BLOCK_SIZE_M * D_HEAD; row += blockDim.x) {
         int m = row / D_HEAD;
         int d = row % D_HEAD;
         if (block_start_m + m < target_seq_len) {
             size_t o_offset = (size_t)batch_idx * q_batch_head_stride + 
                              (block_start_m + m) * query_heads * D_HEAD + 
                              query_head_idx * D_HEAD + d;
             o_ptr[o_offset] = output_buffer[m][d];
         }
     }
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
           constexpr int BLOCK_SIZE_M = 32;
           constexpr int BLOCK_SIZE_N = 32;
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
           constexpr int BLOCK_SIZE_M = 128;
           constexpr int BLOCK_SIZE_N = 128;
           dim3 grid_dim((target_seq_len + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M, query_heads, batch_size);
           dim3 block_dim(128);
           const float scale = 1.0f / sqrt(static_cast<float>(D_HEAD));
           flash_attention_kernel<T, BLOCK_SIZE_M, BLOCK_SIZE_N, D_HEAD><<<grid_dim, block_dim>>>(d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, query_heads, kv_heads, is_causal, scale);
           break;
       }
       case 16: {
           constexpr int D_HEAD = 16;
           constexpr int BLOCK_SIZE_M = 64;
           constexpr int BLOCK_SIZE_N = 64;
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
           constexpr int BLOCK_SIZE_M = 64;
           constexpr int BLOCK_SIZE_N = 64;
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
       case 96: {
           constexpr int D_HEAD = 96;
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
       case 160: {
           constexpr int D_HEAD = 160;
           constexpr int BLOCK_SIZE_M = 8;
           constexpr int BLOCK_SIZE_N = 8;
           dim3 grid_dim((target_seq_len + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M, query_heads, batch_size);
           dim3 block_dim(64);
           const float scale = 1.0f / sqrt(static_cast<float>(D_HEAD));
           flash_attention_kernel<T, BLOCK_SIZE_M, BLOCK_SIZE_N, D_HEAD><<<grid_dim, block_dim>>>(d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, query_heads, kv_heads, is_causal, scale);
           break;
       }
       case 256: {
           constexpr int D_HEAD = 256;
           constexpr int BLOCK_SIZE_M = 8;
           constexpr int BLOCK_SIZE_N = 8;
           dim3 grid_dim((target_seq_len + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M, query_heads, batch_size);
           dim3 block_dim(32);
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
