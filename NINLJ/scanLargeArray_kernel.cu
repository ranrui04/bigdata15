/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */


// Define this to more rigorously avoid bank conflicts, 
// even at the lower (root) levels of the tree
// Note that due to the higher addressing overhead, performance 
// is lower with ZERO_BANK_CONFLICTS enabled.  It is provided
// as an example.
//#define ZERO_BANK_CONFLICTS 
#include "scanLargeArray.cuh"
// 16 banks on G80
#define NUM_BANKS 16
#define LOG_NUM_BANKS 4

/*#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS + (index) >> (2*LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS)
#endif*/

#ifdef COALESCED
	inline __device__
	int CONFLICT_FREE_OFFSET(int index)
	{
		//return ((index) >> LOG_NUM_BANKS + (index) >> (2*LOG_NUM_BANKS));
		return ((index) >> LOG_NUM_BANKS);
	}

	template <bool isNP2>
	__device__ void loadSharedChunkFromMem(int *s_data,
										   const int *g_idata, 
										   int n, int baseIndex,
										   int& ai, int& bi, 
										   int& mem_ai, int& mem_bi, 
										   int& bankOffsetA, int& bankOffsetB)
	{
		int thid = threadIdx.x;
		mem_ai = baseIndex + threadIdx.x;
		mem_bi = mem_ai + blockDim.x;

		ai = thid;
		bi = thid + blockDim.x;

		// compute spacing to avoid bank conflicts
		bankOffsetA = CONFLICT_FREE_OFFSET(ai);
		bankOffsetB = CONFLICT_FREE_OFFSET(bi);

		// Cache the computational window in shared memory
		// pad values beyond n with zeros
		s_data[ai + bankOffsetA] = g_idata[mem_ai]; 
	    
		if (isNP2) // compile-time decision
		{
			s_data[bi + bankOffsetB] = (bi < n) ? g_idata[mem_bi] : 0; 
		}
		else
		{
			s_data[bi + bankOffsetB] = g_idata[mem_bi]; 
		}
	}
#else
	inline __device__
	int CONFLICT_FREE_OFFSET(int index)
	{
		//return ((index) >> LOG_NUM_BANKS + (index) >> (2*LOG_NUM_BANKS));
		//return ((index) >> LOG_NUM_BANKS);

		return 0;
	}

	template <bool isNP2>
	__device__ void loadSharedChunkFromMem(int *s_data,
										   const int *g_idata, 
										   int n, int baseIndex,
										   int& ai, int& bi, 
										   int& mem_ai, int& mem_bi, 
										   int& bankOffsetA, int& bankOffsetB)
	{
		int thid = threadIdx.x;
		mem_ai = baseIndex + 2*threadIdx.x;
		mem_bi = mem_ai + 1;

		ai = 2*thid;
		bi = ai + 1;

		// compute spacing to avoid bank conflicts
		bankOffsetA = CONFLICT_FREE_OFFSET(ai);
		bankOffsetB = CONFLICT_FREE_OFFSET(bi);

		// Cache the computational window in shared memory
		// pad values beyond n with zeros
		s_data[ai + bankOffsetA] = g_idata[mem_ai]; 
	    
		if (isNP2) // compile-time decision
		{
			s_data[bi + bankOffsetB] = (bi < n) ? g_idata[mem_bi] : 0; 
		}
		else
		{
			s_data[bi + bankOffsetB] = g_idata[mem_bi]; 
		}
	}
#endif


template <bool isNP2>
__device__ void storeSharedChunkToMem(int* g_odata, 
                                      const int* s_data,
                                      int n, 
                                      int ai, int bi, 
                                      int mem_ai, int mem_bi,
                                      int bankOffsetA, int bankOffsetB)
{
    __syncthreads();

    // write results to global memory
    g_odata[mem_ai] = s_data[ai + bankOffsetA]; 
    if (isNP2) // compile-time decision
    {
        if (bi < n)
            g_odata[mem_bi] = s_data[bi + bankOffsetB]; 
    }
    else
    {
        g_odata[mem_bi] = s_data[bi + bankOffsetB]; 
    }
}

template <bool storeSum>
__device__ void clearLastElement(int* s_data, 
                                 int *g_blockSums, 
                                 int blockIndex)
{
    if (threadIdx.x == 0)
    {
        int index = (blockDim.x << 1) - 1;
        index += CONFLICT_FREE_OFFSET(index);
        
        if (storeSum) // compile-time decision
        {
            // write this block's total sum to the corresponding index in the blockSums array
            g_blockSums[blockIndex] = s_data[index];
        }

        // zero the last element in the scan so it will propagate back to the front
        s_data[index] = 0;
    }
}





template <bool storeSum>
__device__ void prescanBlock(int *data, int blockIndex, int *blockSums)
{
    int stride = buildSum(data);               // build the sum in place up the tree
    clearLastElement<storeSum>(data, blockSums, 
                               (blockIndex == 0) ? blockIdx.x : blockIndex);
    scanRootToLeaves(data, stride);            // traverse down tree to build the scan 
}

//no shared memory
template <bool storeSum, bool isNP2>
__global__ void prescan(int* d_data,
						int *g_odata, 
                        const int *g_idata, 
                        int *g_blockSums, 
                        int n, 
                        int blockIndex, 
                        int baseIndex, 
						unsigned int sharedMemSize
						)
{
    int ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB;
    //extern __shared__ int s_data[];

	int bx = blockIdx.x;
	int* s_data = d_data + (sharedMemSize/sizeof(int))*bx;

    // load data into shared memory
    loadSharedChunkFromMem<isNP2>(s_data, g_idata, n, 
                                  (baseIndex == 0) ? 
                                  __mul24(blockIdx.x, (blockDim.x << 1)):baseIndex,
                                  ai, bi, mem_ai, mem_bi, 
                                  bankOffsetA, bankOffsetB); 
    // scan the data in each block
    prescanBlock<storeSum>(s_data, blockIndex, g_blockSums); 
    // write results to device memory
    storeSharedChunkToMem<isNP2>(g_odata, s_data, n, 
                                 ai, bi, mem_ai, mem_bi, 
                                 bankOffsetA, bankOffsetB);  
}

//with shared memory
template <bool storeSum, bool isNP2>
__global__ void prescan(int *g_odata, 
                        const int *g_idata, 
                        int *g_blockSums, 
                        int n, 
                        int blockIndex, 
                        int baseIndex
						)
{
    int ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB;
    extern __shared__ int s_data[];

    // load data into shared memory
    loadSharedChunkFromMem<isNP2>(s_data, g_idata, n, 
                                  (baseIndex == 0) ? 
                                  __mul24(blockIdx.x, (blockDim.x << 1)):baseIndex,
                                  ai, bi, mem_ai, mem_bi, 
                                  bankOffsetA, bankOffsetB); 
    // scan the data in each block
    prescanBlock<storeSum>(s_data, blockIndex, g_blockSums); 
    // write results to device memory
    storeSharedChunkToMem<isNP2>(g_odata, s_data, n, 
                                 ai, bi, mem_ai, mem_bi, 
                                 bankOffsetA, bankOffsetB);  
}







