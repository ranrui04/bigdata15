__device__ unsigned int buildSum(int *s_data);
__device__ void scanRootToLeaves(int *s_data, unsigned int stride);
__global__ void uniformAdd(int *g_data, 
                           int *uniforms, 
                           int n, 
                           int blockOffset, 
                           int baseIndex);
