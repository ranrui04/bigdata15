#include "CUDA_ninlj.cuh"
//#include "CSSTree.cuh"
//#include "CUDA_inlj.cuh"
//#include "getMax.cuh"
//#include "CUDA_smj.cuh"
//#include "RadixClusteredHashJoin.cuh"

void testNINLJ(int rLen, int sLen);
/*void testINLJ(int rLen, int sLen);
void testMax(int rLen);
void testSMJ(int rLen, int sLen);
void testMatch(int rLen, int sLen);
void testHJ(int rLen, int sLen);
void testNINLJ_2(Record* h_R, int rLen, Record* h_S, int sLen);
void testINLJ_2(Record* h_R,int rLen, Record* h_S,  int sLen);
void testSMJ_2(Record *h_R, int rLen, Record *h_S, int sLen);
void testHJ_2(Record *h_R, int rLen, Record *h_S, int sLen);
void testSkew(int rLen, int sLen, int mode, double oneRatio);
void testSel(int rLen, int sLen, int mode, float joinSel);
void testMicroJoin(int tL);*/
int testAllJoin(int argc, char ** argv);
