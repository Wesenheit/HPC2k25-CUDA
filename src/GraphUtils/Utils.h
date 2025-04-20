#if !defined(UTILS_H)
#define UTILS_H

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>

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

template<typename T>
double ConvertToRadian(T coord) {
    int deg = static_cast<int>(coord);
    T min = coord - deg;
    return M_PI * (deg + 5.0 * min / 3.0) / 180.0;
}


#endif // UTILS_H
