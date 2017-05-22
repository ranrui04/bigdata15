#include "QP_Utility.cuh"

int gpu_ninlj(Record *d_R, int rLen, Record *d_S, int sLen, Record** d_Rout);
int gpu_ninlj_Constant(Record *d_R, int rLen, Record *d_S, int sLen, Record** d_Rout);
int matchingBlocks(Record *d_R, int rLen, Record *d_S, int sLen, Record** d_match);
