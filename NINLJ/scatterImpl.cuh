#include "QP_Utility.cuh"
void scatterImpl(Record *d_R, int rLen, int *d_loc, Record *d_S, int numThreadPB, int numBlock);
void scatterImpl_forPart(Record *d_R, int rLen, int numPart, int *d_loc, Record *d_S);
void gatherImpl(Record *d_R, int rLen, int *d_loc, Record *d_S, int sLen, int numThreadsPerBlock_x, int numBlock_x);
__global__ void
optScatter( Record *d_R, int delta, int rLen, int *loc, int from, int to, Record *d_S);
__global__ void
optGather( Record *d_R, int delta, int rLen, int *loc, int from, int to, Record *d_S, int sLen);
__global__ void
	optScatter_noCoalesced( Record *d_R, int delta, int rLen, int *loc, int from, int to, Record *d_S);
__global__ void 
	optGather_noCoalesced( Record *d_R, int delta, int rLen, int *loc, int from, int to, Record *d_S, int sLen);
