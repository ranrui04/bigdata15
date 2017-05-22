#ifndef __COMMON_CUH__
#define __COMMON_CUH__

#include <cuda_runtime.h>
#include "QP_Utility.cuh"
#include "scanImpl.cuh"
#include "scatterImpl.cuh"
#include "GPUPrimitive_Def.cu"

#define NUM_RECORDS_R (512*512*4*16)

#define NUM_THREADS_SORT (512)
#define NUM_BLOCKS_X_SORT (NUM_RECORDS_R/NUM_THREADS_SORT)
#define NUM_BLOCKS_Y_SORT (1)

#define REDUCE_SUM (0)
#define REDUCE_MAX (1)
#define REDUCE_MIN (2)
#define REDUCE_AVERAGE (3)

#define SPLIT (4)
#define PARTITION (5)

typedef int4 cmp_type_t;

int TwoPowerN( int n );
void gpuPrint(int *d_output, int numResults, char *notes);
void gpuPrintInterval(int *d_output, int numResults, char *notes);
void gpuPrintInt2(Record *d_output, int numResults, char *notes);
void gpuPrintFloat(float *d_output, int numResults, char *notes);
void validateScan( int* input, int rLen, int* output );
void validateProjection( Record* h_Rin, int rLen, Record* originalProjTable, Record* h_projTable, int pLen );
void validateAggAfterGroupBy( Record *Rin, int rLen, int* startPos, int numGroups, Record* Ragg, int* aggResults, int OPERATOR );
void validateGroupBy( Record* h_Rin, int rLen, Record* h_Rout, int* h_startPos, int numGroup );
void validateFilter( Record* d_Rin, int beginPos, int rLen, 
					Record* Rout, int outSize, int smallKey, int largeKey);
void validateReduce( Record* R, int rLen, unsigned int gpuResult, int OPERATOR );
void validateSort(Record *R, int rLen);
void gpuValidateSort(Record *d_R, int rLen);
void validateSplit(Record *R, int rLen, int numPart);
unsigned int cpu_RSHash(int value, int mask);
void validatePartition( Record* R, int rLen, int numPart );
int get2N( int rLen );
bool is2n(unsigned int i);
void array_startTime(int i);
void array_endTime(char *info,int i);

#endif