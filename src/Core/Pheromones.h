#ifndef PHEROMONES_H
#define PHEROMONES_H

#include <cuda_runtime.h>

#define tile_size 1024

typedef struct {
  int i;
  int j;
  float load;
} Deposits;

__global__ void ConstructDeposits(int *tour, Deposits *array, float *distances,
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
    array[idx * N + num].i = current;
    array[idx * N + num].j = next;
    array[idx * N + num].load = 1.0 / distance;
    current = next;
  }
}

template <typename T>
__device__ void load_local(T *local, T *array, int size_to_load,
                           int num_threads, int offset, int max_size) {
  int index = threadIdx.x;
  while (index < size_to_load && index + offset < max_size) {
    local[index] = array[offset + index];
    index += num_threads;
  }
}
__global__ void DeposePheromones(Deposits *array, float *pheromones, int N) {
  int i = threadIdx.x;
  int j = blockIdx.x;

  __shared__ Deposits local[tile_size];

  for (int idx = 0; idx < N * N; idx += tile_size) {
    load_local(local, array, tile_size, blockDim.x, idx, N * N);
    __syncthreads();
    for (int idx_loc = 0; idx_loc < tile_size && idx + idx_loc < N * N;
         idx_loc++) {
      if ((local[idx_loc].i == i && local[idx_loc].j == j) ||
          (local[idx_loc].i == j && local[idx_loc].j == i)) {
        pheromones[i * N + j] += local[idx_loc].load;
      }
    }
    __syncthreads();
  }
}

__global__ void ReducePheromones(float *pheromones, float evaporate, int N) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = 0; i < N; i++) {
    pheromones[idx * N + i] *= (1 - evaporate);
  }
}

#endif
