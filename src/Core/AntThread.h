#ifndef ANT_THREAD_H
#define ANT_THREAD_H
#include "../GraphUtils/Graph.h"
#include "../GraphUtils/Utils.h"
#include "Common.h"


#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cassert>
#include <vector>

__global__ void TourConstruction_AntThread(float * pheromones, float* distances,curandState* states,int * tours,int N,float alpha, float beta)
{
    /// N - total size of the graph 
    /// phromones - pheromones on the edges [N*N]
    /// distances - distances between the cities [N*N]
    /// states - random states for each thread
    /// tours - the tours for each ant [N*N]
    /// alpha - pheromone importance
    /// beta - distance importance

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int local_idx = threadIdx.x; //local idx in group
    if (idx < N)
    {
    
        extern __shared__ float shared_mem[];
        int local_size = blockDim.x * N;
        float * visited = shared_mem;
        float * selection_prob  = &visited[local_size];
    
        for (int i = 0; i < N; i++)
        {
            visited[local_idx * N + i] = 0.0;
            selection_prob[local_idx * N + i] = 0.0;
        }
        tours[idx * N] = idx; // which city we are in
        visited[local_idx * N + idx] = 1.0; // mark the current city as visited
    
        int current = idx; // we are starting from the current city
    
        for (int num = 1; num < N; num++)
        {
            for (int j = 0; j < N; j++)
            {
                if (visited[local_idx * N + j] > 0)
                {
                    selection_prob[local_idx * N + j] = 0.0;
                }
                else
                {
                    float current_prob = powf(pheromones[current * N + j],alpha) * powf(distances[current * N + j],-beta);
                    selection_prob[local_idx * N + j] = current_prob;
                }

            }
            current = select_prob(&selection_prob[local_idx * N], N, &states[idx]);
            visited[local_idx * N + current] = 1.0;
            tours[idx * N + num] = current; // which city we are in
        }
    }
}


std::pair<float,std::vector<float>> AntThread(Graph & graph, int num_iterations, float alpha, float beta, float evaporate, unsigned long seed)
{
    int threads_per_block = 1024;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    dim3 blocks;
    dim3 threads;
    size_t local_mem_size;
    
    while (true)
    {
        // Memory requirements for this method are enorumous
        // in order to always solve the problem we need to limit the number of threads
        // to do so, let's decrease the number of threads per block until the solution is found
        threads = dim3(threads_per_block);
        blocks = dim3((graph.N + threads.x - 1) / threads.x);
        local_mem_size = threads.x * graph.N * sizeof(float) * 2 ;
        if (prop.sharedMemPerBlock > local_mem_size)
        {
            break;
        }
        else
        {
            threads_per_block /= 2;
            if (threads_per_block == 1 )
            {
                throw std::runtime_error("Too few threads per block");
            }
        }
    }

    curandState* states;
    gpuErrchk(cudaMalloc((void**)&states, graph.N * sizeof(curandState)));
    init_rng<<<blocks, threads>>>(states, seed,graph.N);

    float * pheromones;

    int * tours;

    gpuErrchk(cudaMalloc((void**)&tours, graph.N *graph.N*sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&pheromones, graph.N * graph.N * sizeof(float)));
    set_to_one<<<graph.N,1>>>(pheromones, graph.N);


    /*
        Lets create a graph with the kernel calls
    */
    cudaStream_t stream; 
    cudaStreamCreate(&stream);
    
    cudaGraph_t cuda_graph;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    
    TourConstruction_AntThread<<<blocks, threads, local_mem_size,stream>>>(pheromones,graph.gpu_distances, states, tours,graph.N,alpha,beta);
    DepositPheromones<<<blocks, threads,0,stream>>>(tours,pheromones,graph.gpu_distances,evaporate,graph.N);
    
    cudaStreamEndCapture(stream, &cuda_graph);
    
    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, cuda_graph, NULL, NULL, 0);


    for (int num = 0;num < num_iterations; num++)
    {
        cudaGraphLaunch(graphExec, stream);
    }

    int * tours_cpu;
    tours_cpu = new int[graph.N * graph.N];
    
    gpuErrchk(cudaMemcpy(tours_cpu, tours, graph.N * graph.N * sizeof(int), cudaMemcpyDeviceToHost));

    auto out = get_best_tour(tours_cpu, graph);

    delete[] tours_cpu;
    gpuErrchk(cudaFree(states));
    gpuErrchk(cudaFree(pheromones));
    gpuErrchk(cudaFree(tours));

    return out;
}


#endif
