#ifndef COMMON_H
#define COMMON_H

#include <cuda_runtime.h>

#include <deque>
#include <vector>

__global__ void init_rng(curandState* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}
 
__global__ void set_val(float* arr, float val, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < N; i++)
    {
        arr[idx * N + i] = val;
    }
}

__global__ void preprocess_distances(float* distances, float *distance_processed, float beta,int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int j = 0; j < N; j++)
    {
        if (idx == j)
        {
            distance_processed[idx * N + j] = 0;
        }
        else{
            distance_processed[idx * N + j] = powf(distances[idx * N + j], -beta);
        }
    }
}



__device__ int select_prob(float * prob, int N, curandState * state)
{
    float sum = 0.0;
    for (int i = 0; i < N; i++)
    {
       sum += prob[i];
    }
    float random = curand_uniform(state) * sum;
    float cumulative_sum = 0.0;
    for (int i = 0; i < N; i++)
    {
       cumulative_sum += prob[i];
       if (cumulative_sum >= random)
       {
           return i;
       }
    }
    return -1; // Should not reach here
}

template <typename T, T (*op)(T, T)>
__device__ void ParallelScan(T* arr, T* warps, int N, T missing)
{
    // Thread identification
    int tid = threadIdx.x;
    int lane_id = tid % 32;
    int warp_id = tid / 32;
    unsigned mask = __activemask();
    
    T thread_val = (tid < N) ? arr[tid] : missing;
    T original_val = thread_val;
    
    T warp_sum = thread_val;
    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        T temp = __shfl_up_sync(mask, warp_sum, offset);
        if (lane_id >= offset) {
            warp_sum = op(temp, warp_sum);
        }
    }
    
    if (lane_id == 31) {
        warps[warp_id] = warp_sum;
    }
    __syncthreads();
    
    if (warp_id == 0 && lane_id < (N + 31) / 32) {
        T temp_val = warps[lane_id];
        
        T scan_val = temp_val;
        for (int i = 0; i < lane_id; i++) {
            T prev_val = warps[i];
            scan_val = op(prev_val, scan_val);
        }
        
        warps[lane_id] = scan_val;
    }
    __syncthreads();
    
    T scan_result;
    if (warp_id == 0) {
        scan_result = warp_sum;
    } else {
        T warp_prefix = warps[warp_id - 1];   
        scan_result = op(warp_prefix, warp_sum);
    }    
    if (tid < N) {
        arr[tid] = scan_result;
    }
}


template <typename T, T (*op)(T, T)>
__device__ void ParallelReduce(T * arr, T * warps, int N, T missing)
{
    int j = threadIdx.x;
    int lane_id = j % 32;
    int warp_id = j / 32;
    unsigned mask = __activemask();

    T value = arr[j];

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
    {
        value = op(__shfl_down_sync(mask, value, offset),value);
    }
    __syncthreads();

    if (lane_id == 0)
    {
        warps[warp_id] = value;
    }
    __syncthreads();

    if (warp_id == 0)
    {
        value = (j < (N / 32)) ? warps[lane_id] : missing;
    
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
        {
            value = op(__shfl_down_sync(mask, value, offset),value);
        }
        warps[lane_id] = value; 
    }
}

std::pair<float,std::vector<int>> get_best_tour(int* &tours_cpu, Graph &graph)
{
    float minimal_distance = MAXFLOAT;
    int idx_of_best_tour = 0;
    for (int idx = 0; idx < graph.N; idx++)
    {
        float current_distance = 0;
        int current = tours_cpu[idx * graph.N];
        for (int num = 0;num < graph.N-1;num++)
        {
            int next = tours_cpu[idx * graph.N + num + 1];
            current_distance += graph.distances[current * graph.N + next];
            current = next;
        }

        if (current_distance < minimal_distance)
        {
            minimal_distance = current_distance;
            idx_of_best_tour = idx;
        }
    }
    std::deque<int> best_tour;
    for (int num = 0;num < graph.N;num++)
    {
        best_tour.push_back(tours_cpu[idx_of_best_tour * graph.N + num]);
    }
    while (best_tour.front() != 0)
    {
        int first = best_tour.front();
        best_tour.pop_front();
        best_tour.push_back(first);
    }
    std::vector<int> out;
    for (auto value : best_tour)
    {
        out.push_back(value);
    }
    return std::make_pair(minimal_distance, out);
}


#endif
