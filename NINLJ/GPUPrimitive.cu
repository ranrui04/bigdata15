

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
//#include <cutil.h>
// includes CUDA
#include <cuda_runtime.h>

// includes, kernels
//#include <TestAll.cu>
#include "TestJoin.cuh"
//#include "GPUDB_AccessMethod.cuh"
//#include "GPUDB_Operator.cuh"
//#include "TestAll.cuh"



////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
	cudaSetDevice(0);
	//int numPart=32;
	//printf("%d",1<<(log2((int)(SHARED_MEMORY_PER_PROCESSOR/(numPart*sizeof(int))))));
	//printf("%d, %d", '\0', 'A'); 
   
	//testAllPrimitive(argc, argv);
	testAllJoin(argc, argv);
	//test_Operators(argc,argv);
	//test_AccessMethods(argc,argv);
	
	//testINLJ(1024*1024*256, 1024*1024*256);

	//testHashSearch(1024*1024, 1024*1024);
	//testTreeSearch(1024*1024, 1024*1024);

	//testHJ(1000*1000, 1000*1000);
	//testHJ(10*1000*1000, 10*1000*1000);
	//testHJ(15*1000*1000, 15*1000*1000);	
	//testRadixSort(1024*1024*8);
	//system("pause");
	//testINLJ( 1024*1024*256, 1024*1024*256 );
	//system("pause");
	//testNINLJ( 1*1024*1024, 1*1024*1024);
	//system("pause");
	//testSMJ(1024*1024*128, 1024*1024*128);
	/*int rLen = 1024*1024*8;
	int numThread = 256;
	int numBlock = 256;
	testAggAfterGroupByImpl( rLen, REDUCE_AVERAGE, numThread, numBlock );*/
	

	//testProjection( 1024*1024*16, 1024*1024*16*0.01, 128, 128 );

	/*int rLen = 1024*1024;
	int numPart = 8;
	testPartition(rLen, numPart );*/

	//testRadixSort( 1024*1024*1 );

	//testFilterImpl( 1024*1024*16 );

	//testGroupByImpl( 1024*1024*16 );

	//testReduceImpl( 1024*1024*16, REDUCE_MAX);

	//testMapImpl( 1024*1024*16, 512, 1024);

	//testGather( 1024*1024*16 );
	//testScatter( 1024*1024*16 );

	//testSplit(1024*1024*16, 64, 64, 1024);

	//testSelection( 1024*1024*16 );

	//testScan( 1024*1024*16 );

	//testQSort( 1024*1024*16 );
	//system("pause");
	return 0;
}


