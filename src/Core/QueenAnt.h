#ifndef QUEEN_ANT_H
#define QUEEN_ANT_H

#include "Common.h"

__global__ void TourConstruction_QueenAnt(float * pheromones, float* distances,curandState* states,int * tours,int N,float alpha, float beta)
{
    /// N - total size of the graph 
    /// phromones - pheromones on the edges [N*N]
    /// distances - distances between the cities [N*N]
    /// states - random states for each thread
    /// tours - the tours for each ant [N*N]
    /// alpha - pheromone importance
    /// beta - distance importance

    int idx = blockIdx.x;
    
    extern __shared__ float shared_mem[];
    int local_size = blockDim.x;
    bool * visited = (bool * ) shared_mem;
    float * selection_prob  = (float *) visited + local_size;   
    int * global_current = (int*) selection_prob + local_size; // global current city
    // not very elegant but works, we are just casting float* to int*

    int j = threadIdx.x;
    
    visited[j] = 0;
    selection_prob[j] = 0.0;
    int current = idx;
    if (j == 0)
    {
        visited[idx] = 1;
        tours[idx * N] = idx;
    }
    for (int num = 1; num < N; num++)
    {
        float current_prob = powf(pheromones[current * N + j], alpha) * powf(distances[current * N + j], -beta);

        selection_prob[j] = (visited[j] > 0) ? 0.0 : current_prob; // only if not visited

        __syncthreads();
        if (j == 0) //only "master ant" will select the next city
        {
            *global_current = select_prob(selection_prob, N, &states[idx]);
            visited[current] = 1;
            tours[idx * N + num] = *global_current; // which city we are in
        }
        __syncthreads();
        current = *global_current; // save global current city to each worker ant
    }
}



std::pair<float,std::vector<float>> QueenAnt(Graph & graph, int num_iterations, float alpha, float beta, float evaporate, unsigned long seed)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    size_t local_mem_size;
    local_mem_size = (graph.N * sizeof(float) + graph.N * sizeof(bool) + sizeof(int)); // two arrays of size N
    
    assert(prop.sharedMemPerBlock > local_mem_size);



    curandState* states;
    gpuErrchk(cudaMalloc((void**)&states, graph.N * sizeof(curandState)));
    init_rng<<<1, graph.N>>>(states, seed);

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
    
    TourConstruction_QueenAnt<<<graph.N, graph.N, local_mem_size,stream>>>(pheromones,graph.gpu_distances, states, tours,graph.N,alpha,beta);
    DepositPheromones<<<1, graph.N,0,stream>>>(tours,pheromones,graph.gpu_distances,evaporate,graph.N);
    
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

    return out;
}

#endif