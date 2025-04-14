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

__device__ float add_op(float a, float b) {
   return a + b;
}

#endif // UTILS_H
