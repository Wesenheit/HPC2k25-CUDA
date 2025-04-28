#ifndef QUEEN_ANT_H
#define QUEEN_ANT_H

#include "Common.h"
#include "Pheromones.h"
#include <cstdint>

__global__ void TourConstruction_QueenAntOptimized(float *pheromones,
                                                   float *distances_processed,
                                                   curandState *states,
                                                   int *tours, float alpha,
                                                   int N) {
  /// N - total size of the graph
  /// pheromones - pheromones on the edges [N*N]
  /// distances - distances between the cities [N*N]
  /// states - random states for each thread
  /// tours - the tours for each ant [N*N]
  /// alpha - pheromone importance
  int biggest_aligned_size = blockDim.x;
  int idx = blockIdx.x;

  extern __shared__ float shared_mem[];
  float *selection_prob = (float *)shared_mem;
  float *warp_sums =
      (float *)selection_prob + biggest_aligned_size;  // used for reduction
  int *global_current = (int *)warp_sums + 32;         // global current city
  float *number_to_find = (float *)global_current + 1; // used for reduction
  int16_t *visited = (int16_t *)global_current + 4;
  // not very elegant but works

  int *selection_prob_int = (int *)selection_prob;
  int *warp_sums_int = (int *)warp_sums;
  // int versions of the same storage, used for parallel reduce

  int j = threadIdx.x;
  unsigned mask = __activemask();

  int current = idx;
  visited[j] = 0;
  selection_prob[j] = 0.0;
  if (j == idx) {
    tours[idx * N] = current;
    visited[idx] = 1;
  }
  __syncthreads();

  for (int num = 1; num < N; num++) {
    float current_prob;
    // Step 1: compute probabilities
    if (j < N) {
      current_prob = __powf(pheromones[current * N + j], alpha) *
                     distances_processed[current * N + j];
    } else {
      current_prob = 0.0;
    }

    selection_prob[j] =
        (visited[j] > 0) ? 0.0 : current_prob; // only if not visited

    __syncthreads();

    // Step 2: proceed with the selection using parallel scan
    ParallelScan<float, add_op>(selection_prob, warp_sums, biggest_aligned_size,
                                0.0);

    __syncthreads();
    // Step 3: select the random number
    if (j == 0) {
      float random = curand_uniform(&states[idx]) * selection_prob[N - 1];
      *number_to_find = random;
    }

    __syncthreads();
    // Step 4: Find minimal index that is bigger than threshold using parallel
    // reduce
    float random = *number_to_find;
    selection_prob_int[j] =
        (selection_prob[j] >= random) ? j : biggest_aligned_size;

    __syncthreads();

    ParallelReduce<int, min>(selection_prob_int, warp_sums_int,
                             biggest_aligned_size, biggest_aligned_size);

    __syncthreads();
    // Step 5: select next city
    if (j == 0) {
      current = warp_sums_int[0];
      tours[idx * N + num] = current; // which city we are in
      *global_current = current;
      visited[current] = 1;
    }
    __syncthreads();
    current = *global_current; // save global current city to each worker ant
  }
}

std::pair<float, std::vector<int>> QueenAnt(Graph &graph, int num_iterations,
                                            float alpha, float beta,
                                            float evaporate,
                                            unsigned long seed) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  size_t local_mem_size;
  int biggest_aligned_size = (graph.N + 31) & ~31;
  local_mem_size =
      (biggest_aligned_size * sizeof(float) +
       biggest_aligned_size * sizeof(uint16_t) + 32 * sizeof(float) +
       4 * sizeof(int)); // two arrays of size N

  assert(prop.sharedMemPerBlock > local_mem_size);

  curandState *states;
  gpuErrchk(cudaMalloc((void **)&states, graph.N * sizeof(curandState)));
  init_rng<<<1, graph.N>>>(states, seed);

  float *pheromones;
  float *distances_processed;
  int *tours;
  Deposits *deposits;

  gpuErrchk(
      cudaMalloc((void **)&deposits, graph.N * graph.N * sizeof(Deposits)));
  gpuErrchk(cudaMalloc((void **)&tours, graph.N * graph.N * sizeof(int)));
  gpuErrchk(
      cudaMalloc((void **)&pheromones, graph.N * graph.N * sizeof(float)));
  gpuErrchk(cudaMalloc((void **)&distances_processed,
                       graph.N * graph.N * sizeof(float)));
  set_val<<<1, graph.N>>>(pheromones, 1 / graph.nearest_neigh(), graph.N);
  preprocess_distances<<<1, graph.N>>>(graph.gpu_distances, distances_processed,
                                       beta, graph.N);

  size_t shared_tile_size = FindTileSize(prop);
  /*
      Lets create a graph with the kernel calls
  */
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaGraph_t cuda_graph;
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

  TourConstruction_QueenAntOptimized<<<graph.N, biggest_aligned_size,
                                       local_mem_size, stream>>>(
      pheromones, distances_processed, states, tours, alpha, graph.N);
  ReducePheromones<<<1, graph.N, 0, stream>>>(pheromones, evaporate, graph.N);
  ConstructDeposits<<<1, graph.N, 0, stream>>>(tours, deposits,
                                               graph.gpu_distances, graph.N);
  DepositPheromones<<<graph.N, graph.N, shared_tile_size * sizeof(Deposits), stream>>>(deposits, pheromones,
                                                    graph.N, shared_tile_size);

  cudaStreamEndCapture(stream, &cuda_graph);

  cudaGraphExec_t graphExec;
  cudaGraphInstantiate(&graphExec, cuda_graph, NULL, NULL, 0);

  for (int num = 0; num < num_iterations; num++) {
    cudaGraphLaunch(graphExec, stream);
  }

  cudaStreamSynchronize(stream);
  int *tours_cpu;

  tours_cpu = new int[graph.N * graph.N];

  gpuErrchk(cudaMemcpy(tours_cpu, tours, graph.N * graph.N * sizeof(int),
                       cudaMemcpyDeviceToHost));

  auto out = get_best_tour(tours_cpu, graph);

  delete[] tours_cpu;
  gpuErrchk(cudaFree(states));
  gpuErrchk(cudaFree(distances_processed));
  gpuErrchk(cudaFree(pheromones));
  gpuErrchk(cudaFree(tours));
  gpuErrchk(cudaFree(deposits));

  return out;
}

#endif
