#include "GPUPrimitive_Def.cu"

inline bool 
isPowerOfTwo(int n);
inline int 
floorPow2(int n);
#define BLOCK_SIZE 256
void preallocBlockSums(unsigned int maxNumElements);
void deallocBlockSums();
void saven_initialPrefixSum(unsigned int maxNumElements);
void prescanArrayRecursive(int *outArray, 
                           const int *inArray, 
                           int numElements, 
                           int level);
void prescanArray(int *outArray, int *inArray, int numElements);
extern int** g_scanBlockSums;
extern unsigned int g_numEltsAllocated;
extern unsigned int g_numLevelsAllocated;