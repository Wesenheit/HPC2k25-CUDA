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
    float * visited = shared_mem;
    float * selection_prob  = visited + local_size;   
    int * global_current = (int*)(selection_prob + local_size); // global current city

    int j = threadIdx.x;
    
    visited[j] = 0.0;
    selection_prob[j] = 0.0;
    int current = idx;;
    if (j == 0)
    {
        visited[idx] = 1.0;
        tours[idx * N] = idx;
    }
    for (int num = 1; num < N; num++)
    {
        
        if (visited[j] > 0)
        {
            selection_prob[j] = 0.0;
        }
        else
        {

            float current_prob = powf(pheromones[current * N + j],alpha) * powf(distances[current * N + j], -beta);
            selection_prob[j] = current_prob;
        }
        __syncthreads();
        if (j == 0) //only "master ant" will select the next city
        {
            *global_current = select_prob(selection_prob, N, &states[idx]);
            visited[current] = 1.0;
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
    dim3 blocks_T = dim3(graph.N);
    dim3 threads_T = dim3(graph.N);
    size_t local_mem_size;
    local_mem_size = (graph.N * 2 + 1) * sizeof(float); // two arrays of size N
    
    assert(prop.sharedMemPerBlock > local_mem_size);


    int threads_per_block = 32;
    dim3 threads_P = dim3(threads_per_block);
    dim3 blocks_P = dim3((graph.N + threads_P.x - 1) / threads_P.x);


    curandState* states;
    gpuErrchk(cudaMalloc((void**)&states, graph.N * sizeof(curandState)));
    init_rng<<<1, graph.N>>>(states, seed,graph.N);

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
    
    TourConstruction_QueenAnt<<<blocks_T, threads_T, local_mem_size,stream>>>(pheromones,graph.gpu_distances, states, tours,graph.N,alpha,beta);
    DepositPheromones<<<blocks_P, threads_P,0,stream>>>(tours,pheromones,graph.gpu_distances,evaporate,graph.N);
    
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