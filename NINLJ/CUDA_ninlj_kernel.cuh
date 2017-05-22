#include "QP_Utility.cuh"
#include <cuda_runtime.h>

__global__ void
gpuNLJ_kernel(int* d_temp, Record *d_R, Record *d_S, int sStart, int rLen, int sLen, int *d_n);
__global__ void
write(Record *d_R, Record *d_S,  int sStart, int rLen, int sLen, int *d_sum, Record *output);
__global__ void
matchCount_kernel(Record *R, Record *S, int sStart, int rLen, int sLen, int *d_n);
__global__ void
matchWrite_kernel(Record *R, Record *S,  int sStart, int rLen, int sLen, int *d_sum, Record *output);
__global__ void
gpuNLJ_Constant_kernel(Record *d_R, Record *d_S, int sStart, int rLen, int sLen, int *d_n);
__global__ void
write_Constant_kernel(Record *d_R, Record *d_S,  int sStart, int rLen, int sLen, int *d_sum, Record *output);
