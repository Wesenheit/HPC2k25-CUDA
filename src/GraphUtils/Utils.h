#if !defined(UTILS_H)
#define UTILS_H

#include <cuda_runtime.h>
#include <curand_kernel.h>

//CURAND routines

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void init_rng(curandState* states, unsigned long seed, int n) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   curand_init(seed, idx, 0, &states[idx]);
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



#endif // UTILS_H
