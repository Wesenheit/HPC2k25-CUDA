#ifndef PHEROMONES_H
#define PHEROMONES_H

#include <cuda_runtime.h>

#define tile 1024

size_t FindTileSize(cudaDeviceProp &prop) {
  size_t size =  prop.sharedMemPerBlock / (sizeof(int)*3);
  return size;
};


__global__ void ConstructDeposits(int *tour, int *array_idx1, int *array_idx2, float *array_load, float *distances,
                                  int N) {
  float distance = 0.;
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int current = tour[idx * N];
  for (int num = 0; num < N - 1; num++) {
    int next = tour[idx * N + num + 1];
    distance += distances[current * N + next];
    current = next;
  }
  current = tour[idx * N];
  for (int num = 0; num < N - 1; num++) {
    int next = tour[idx * N + num + 1];
    array_idx1[idx * N + num] = current;
    array_idx2[idx * N + num] = next;
    array_load[idx * N + num] = 1.0 / distance;
    current = next;
  }
}

template <typename T>
__device__ void load_local(T *local,const T *array, int size_to_load,
                           int num_threads, int offset, int max_size) {
  int index = threadIdx.x;
  while (index < size_to_load && index + offset < max_size) {
    local[index] = __ldg(&array[offset + index]);
    index += num_threads;
  }
}
__global__ void DepositPheromones(const int *array_idx1,const int * array_idx2,const float * array_load, float *pheromones, int N) {
  int j = threadIdx.x;
  int i = blockIdx.x;
  float val = 0;
  __shared__  int local_idx1[tile];
  __shared__  int local_idx2[tile];
  __shared__  float local_load[tile];

  for (int idx = 0; idx < N * N; idx += tile) {
    load_local(local_idx1, array_idx1, tile, blockDim.x, idx, N * N);
    load_local(local_idx2, array_idx2, tile, blockDim.x, idx, N * N);
    load_local(local_load, array_load, tile, blockDim.x, idx, N * N);
    __syncthreads();
    for (int idx_loc = 0; idx_loc < tile && idx + idx_loc < N * N;
         idx_loc++) {
      if ((local_idx1[idx_loc] == i && local_idx2[idx_loc] == j) ||
          (local_idx1[idx_loc] == j && local_idx2[idx_loc] == i)) {
        val += local_load[idx_loc];
      }
    }
    __syncthreads();
  }
  pheromones[i * N + j] = val;
}

__global__ void ReducePheromones(float *pheromones, float evaporate, int N) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = 0; i < N; i++) {
    pheromones[idx * N + i] *= (1 - evaporate);
  }
}

#endif
