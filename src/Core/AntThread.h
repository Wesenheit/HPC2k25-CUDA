#ifndef ANT_THREAD_H
#define ANT_THREAD_H
#include "../GraphUtils/Graph.h"
#include "../GraphUtils/Utils.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cassert>
#include <vector>

__global__ void TourConstruction(float * pheromones, float* distances,curandState* states,int * tours,int N,float alpha, float beta)
{
    /// N - total size of the graph 
    /// 
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int local_idx = threadIdx.x; //local idx in group
    float sum_prob = 0.0;
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
    
        int current = idx; // we are starting from the current city
    
        for (int num = 0; num < N; num++)
        {
            for (int j = 0; j < N; j++)
            {
                if (visited[local_idx * N + j] > 0)
                {
                    selection_prob[local_idx * N + j] = 0.0;
                }
                else
                {
                    if (distances[current * N + j] == 0) // no path between the cities
                    {
                        selection_prob[local_idx * N + j] = 0.0;
                    }
                    else
                    {
                        float current_prob = powf(pheromones[current * N + j],alpha) * powf(1.0 / distances[current * N + j], beta);
                        selection_prob[local_idx * N + j] = current_prob;
                        sum_prob += current_prob;;
                    }
                }

            }
            current = select_prob(&selection_prob[local_idx * N], N, &states[idx]);
            visited[local_idx * N + current] = 1.0;
            tours[idx * N + num] = current; // which city we are in
        }
    }
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


std::pair<float,std::vector<float>> AntThread(Graph & graph, int num_iterations, float alpha, float beta, float evaporate, unsigned long seed)
{
    dim3 threads(16);
    dim3 blocks((graph.N + threads.x - 1) / threads.x);
    size_t local_mem_size = threads.x * graph.N * sizeof(float) * 2 ;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    assert(prop.sharedMemPerBlock > local_mem_size);

    curandState* states;
    gpuErrchk(cudaMalloc((void**)&states, graph.N * sizeof(curandState)));
    init_rng<<<blocks, threads>>>(states, seed, threads.x);

    float * pheromones;

    int * tours;

    gpuErrchk(cudaMalloc((void**)&tours, graph.N *graph.N*sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&pheromones, graph.N * graph.N * sizeof(float)));
    gpuErrchk(cudaMemset(pheromones, 1, graph.N * graph.N * sizeof(float)));


    /*
        Lets create a graph with the kernel calls
    */
    cudaStream_t stream; 
    cudaStreamCreate(&stream);
    
    cudaGraph_t cuda_graph;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    
    TourConstruction<<<blocks, threads, local_mem_size,stream>>>(pheromones,graph.gpu_distances, states, tours,graph.N,alpha,beta);
    DepositPheromones<<<blocks, threads,0,stream>>>(tours,pheromones,graph.gpu_distances,evaporate,graph.N);
    
    cudaStreamEndCapture(stream, &cuda_graph);
    
    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, cuda_graph, NULL, NULL, 0);


    for (int num = 0;num < num_iterations; num++)
    {
        cudaGraphLaunch(graphExec, stream);
    }

    int * tours_cpu;

    int idx_of_best_tour = 0;
    tours_cpu = new int[graph.N * graph.N];
    
    gpuErrchk(cudaMemcpy(tours_cpu, tours, graph.N * graph.N * sizeof(int), cudaMemcpyDeviceToHost));

    float minimal_distance = MAXFLOAT;

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

    delete[] tours_cpu;
    gpuErrchk(cudaFree(states));
    gpuErrchk(cudaFree(pheromones));
    gpuErrchk(cudaFree(tours));

    return std::make_pair(minimal_distance, best_tour);
}


#endif
