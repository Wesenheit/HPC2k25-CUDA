#ifndef COMMON_H
#define COMMON_H

#include <cuda_runtime.h>

#include <vector>

__global__ void init_rng(curandState* states, unsigned long seed, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    curand_init(seed, idx, 0, &states[idx]);
}
 
__global__ void set_to_one(float* arr, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < N; i++)
    {
        arr[idx * N + i] = 1.0f;
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


__global__ void DepositPheromones(int * tour, float* pheromones, float* distances,float evaporate,int N)
{
    float distance = 0;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N)
    {  
        int current = tour[idx * N];
        for (int num = 0;num < N-1;num++)
        {
            int next = tour[idx * N + num + 1];
            distance += distances[current * N + next];
            current = next;
        }
        for (int num = 0;num < N;num++)
        {
            pheromones[idx * N + num] *= (1.0 - evaporate); //reduce the pheromones
        }
        
        current = tour[idx * N];
        for (int num = 0;num < N-1;num++)
        {
            int next = tour[idx * N + num + 1];
            pheromones[current * N + next] += 1/distances[current * N + next];
            current = next;
        }
    }
}


std::pair<float,std::vector<float>> get_best_tour(int* &tours_cpu, Graph &graph)
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
    std::vector<float> best_tour(graph.N);
    for (int num = 0;num < graph.N;num++)
    {
        best_tour[num] = tours_cpu[idx_of_best_tour * graph.N + num];
    }
    return std::make_pair(minimal_distance, best_tour);
}


#endif // COMMON_H)
