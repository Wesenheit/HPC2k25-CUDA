#ifndef ANT_THREAD_H
#define ANT_THREAD_H
#include "../GraphUtils/Graph.h"
#include "../GraphUtils/Utils.h"
#include "Common.h"


#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cassert>
#include <vector>

__global__ void TourConstruction_AntThread(float * pheromones, float* distances_processed,curandState* states,int * tours,int N,float alpha)
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

                float current_prob = powf(pheromones[current * N + j],alpha) * distances_processed[current * N + j];
                selection_prob[local_idx * N + j] = (visited[local_idx * N + j] > 0) ? 0.0 : current_prob; // only if not visited
            }
            current = select_prob(&selection_prob[local_idx * N], N, &states[idx]);
            visited[local_idx * N + current] = 1.0;
            tours[idx * N + num] = current; // which city we are in
        }
    }
}


std::pair<float,std::vector<int>> AntThread(Graph & graph, int num_iterations, float alpha, float beta, float evaporate, unsigned long seed)
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
    init_rng<<<1, graph.N>>>(states, seed);

    float * pheromones;
    float * distances_processed;
    int * tours;

    gpuErrchk(cudaMalloc((void**)&tours, graph.N *graph.N*sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&pheromones, graph.N * graph.N * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&distances_processed, graph.N * graph.N * sizeof(float)));

    set_val<<<graph.N,1>>>(pheromones, 1/(graph.nearest_neigh() * graph.N),graph.N);
    preprocess_distances<<<1,graph.N>>>(graph.gpu_distances, distances_processed, beta, graph.N);


    /*
        Lets create a graph with the kernel calls
    */
    cudaStream_t stream; 
    cudaStreamCreate(&stream);
    
    cudaGraph_t cuda_graph;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    
    TourConstruction_AntThread<<<blocks, threads, local_mem_size,stream>>>(pheromones,distances_processed, states, tours,graph.N,alpha);
    DepositPheromones<<<1, graph.N,0,stream>>>(tours,pheromones,graph.gpu_distances,evaporate,graph.N);
    // Deposit Pheromones is called with the single block! The reason is that AtomicAdd is used.
    // It is atomic only withing the block, it would not work on many blocks.
    
    cudaStreamEndCapture(stream, &cuda_graph);
    
    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, cuda_graph, NULL, NULL, 0);


    for (int num = 0;num < num_iterations; num++)
    {
        cudaGraphLaunch(graphExec, stream);
    }
    cudaStreamSynchronize(stream);

    int * tours_cpu;
    tours_cpu = new int[graph.N * graph.N];
    
    gpuErrchk(cudaMemcpy(tours_cpu, tours, graph.N * graph.N * sizeof(int), cudaMemcpyDeviceToHost));

    auto out = get_best_tour(tours_cpu, graph);

    delete[] tours_cpu;
    gpuErrchk(cudaFree(states));
    gpuErrchk(cudaFree(pheromones));
    gpuErrchk(cudaFree(tours));
    gpuErrchk(cudaFree(distances_processed));

    return out;
}


#endif
