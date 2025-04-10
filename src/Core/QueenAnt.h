#ifndef QUEEN_ANT_H
#define QUEEN_ANT_H

#include "Common.h"

__global__ void TourConstruction_QueenAnt(float * pheromones, float* distances_processed,curandState* states,int * tours,float alpha, int biggest_aligned_size)
{
    /// N - total size of the graph 
    /// phromones - pheromones on the edges [N*N]
    /// distances - distances between the cities [N*N]
    /// states - random states for each thread
    /// tours - the tours for each ant [N*N]
    /// alpha - pheromone importance
    /// beta - distance importance
    int N = blockDim.x;
    int idx = blockIdx.x;
    
    extern __shared__ float shared_mem[];
    int * visited = (int * ) shared_mem;
    float * selection_prob  = (float *) visited + biggest_aligned_size;   
    int * global_current = (int*) selection_prob + biggest_aligned_size; // global current city
    // not very elegant but works, we are just casting float* to int*

    int j = threadIdx.x;
    
    int current = idx;
    visited[j] = 0;
    selection_prob[j] = 0.0;
    if (j == idx)
    {
        tours[idx * N] = current;
        visited[idx] = 1;

    }

    __syncthreads();
    for (int num = 1; num < N; num++)
    {
        float current_prob = powf(pheromones[current * N + j], alpha) * distances_processed[current * N + j];

        selection_prob[j] = (visited[j] > 0) ? 0.0 : current_prob; // only if not visited

        __syncthreads();
        if (j == 0) //only "master ant" will select the next city
        {
            current = select_prob(selection_prob, N, &states[idx]);
            tours[idx * N + num] = current; // which city we are in
            *global_current = current;
            visited[current] = 1;
        }
        __syncthreads();
        current = *global_current; // save global current city to each worker ant
    }
}



std::pair<float,std::vector<int>> QueenAnt(Graph & graph, int num_iterations, float alpha, float beta, float evaporate, unsigned long seed)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    size_t local_mem_size;
    int biggest_aligned_size = (graph.N / 4 + 1 )*4;
    local_mem_size = (biggest_aligned_size * sizeof(float) + biggest_aligned_size * sizeof(int) + 4*sizeof(int)); // two arrays of size N
    
    assert(prop.sharedMemPerBlock > local_mem_size);



    curandState* states;
    gpuErrchk(cudaMalloc((void**)&states, graph.N * sizeof(curandState)));
    init_rng<<<1, graph.N>>>(states, seed);

    float * pheromones;
    float * distances_processed;
    int * tours;

    gpuErrchk(cudaMalloc((void**)&tours, graph.N *graph.N*sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&pheromones, graph.N * graph.N * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&distances_processed, graph.N * graph.N * sizeof(float)));
    set_val<<<graph.N,1>>>(pheromones, 1/graph.nearest_neigh(),graph.N);
    preprocess_distances<<<1,graph.N>>>(graph.gpu_distances, distances_processed, beta, graph.N);

    /*
        Lets create a graph with the kernel calls
    */
    cudaStream_t stream; 
    cudaStreamCreate(&stream);
    
    cudaGraph_t cuda_graph;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    
    TourConstruction_QueenAnt<<<graph.N, graph.N, local_mem_size,stream>>>(pheromones,distances_processed, states, tours,alpha,biggest_aligned_size);
    //DepositPheromones<<<1, graph.N,0,stream>>>(tours,pheromones,graph.gpu_distances,evaporate,graph.N);
    
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
    gpuErrchk(cudaFree(distances_processed));
    gpuErrchk(cudaFree(pheromones));
    gpuErrchk(cudaFree(tours));

    return out;
}

#endif
