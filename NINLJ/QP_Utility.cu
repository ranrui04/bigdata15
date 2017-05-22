#ifndef QP_UTILITY_CU
#define QP_UTILITY_CU
#include "stdlib.h"
#include "stdio.h"
#include "math.h"
#include "time.h"
#include "assert.h"
//#include <cuda_runtime.h>
//#include <cutil.h>

#include "QP_Utility.cuh"
#include <helper_functions.h>

//typedef int2 Record;

#define RAND_RANGE(N) ((double)rand()/((double)RAND_MAX + 1)*(N))

int seeded = 0;
unsigned int seedValue;

void seed_generator(unsigned int seed)
{
    srand(seed);
    seedValue = seed;
    seeded = 1;
}

void check_seed()
{
    if(!seeded)
    {
	seedValue = time(NULL);
	srand(seedValue);
	seeded = 1;
    }
}

void knuth_shuffle(Record *relation, int num_tuples)
{
    int i;
    for (i = num_tuples-1; i>0; i--)
    {
	int j = RAND_RANGE(i);
	int k_tmp = relation[i].y;
	relation[i].y = relation[j].y;
	relation[j].y = k_tmp;
    }
}

void random_unique_gen(Record *rel, int num_tuples)
{
    int i;
    for (i = 0; i < num_tuples; i++)
	rel[i].x = rel[i].y = (i+1);
    knuth_shuffle(rel, num_tuples);
}

int create_relation_pk(Record *relation, int num_tuples)
{
    check_seed();
    random_unique_gen(relation, num_tuples);
    return 0;
}

int compare (const void * a, const void * b)
{
  return ( ((Record*)a)->y - ((Record*)b)->y );
}
void randomize(Record *R, int rLen, int times)
{
	int i=0;
	int temp=0;
	int from=0;
	int to=0;
	srand(times);
	const int offset=(1<<15)-1;
	for(i=0;i<times;i++)
	{
		from=((((rand()& offset)<<15)+(rand()&1))+(rand()<<1)+(rand()&1))%rLen;
		to=((((rand()& offset)<<15)+(rand()&1))+(rand()<<1)+(rand()&1))%rLen;
		temp=R[from].y;
		R[from].y=R[to].y;
		R[to].y=temp;		
	}
	
}

void int_randomize(int *R, int rLen, int times)
{
	int i=0;
	int temp=0;
	int from=0;
	int to=0;
	srand(times);
	const int offset=(1<<15)-1;
	for(i=0;i<times;i++)
	{
		from=((((rand()& offset)<<15)+(rand()&1))+(rand()<<1)+(rand()&1))%rLen;
		to=((((rand()& offset)<<15)+(rand()&1))+(rand()<<1)+(rand()&1))%rLen;
		temp=R[from];
		R[from]=R[to];
		R[to]=temp;		
	}
	
}

/************************************************************************/
/* This function generates <rLen> random tuples; maybe duplicated. 
/************************************************************************/
void generateRand(Record *R, int maxmax, int rLen, int seed)
{
	int i=0;
	const int offset=(1<<15)-1;
	srand(seed);
	for(i=0;i<rLen;i++)
	{
		R[i].y=((((rand()& offset)<<15)+(rand()&1))+(rand()<<1)+(rand()&1))%maxmax;
		
		//R[i].x=i+1;
		R[i].x=i;
	}
}

void generateRand1(Record *R, int maxmax, int rLen, int seed)
{
	int i=0;
	const int offset=(1<<15)-1;
	srand(seed);
	for(i=0;i<rLen;i++)
	{
		R[i].y=((((rand()& offset)<<15)+(rand()&1))+(rand()<<1)+(rand()&1))%maxmax;
		
		//R[i].x=i+1;
		R[i].x=i;
	}
}
/************************************************************************/
/* This function generates <rLen> random tuples; maybe duplicated. 
/************************************************************************/
void generateRandInt(int *R, int max, int rLen, int seed)
{
	int i=0;
	const int offset=(1<<15)-1;
	srand(seed);
	for(i=0;i<rLen;i++)
	{
		R[i]=((((rand()& offset)<<15)+(rand()&1))+(rand()<<1)+(rand()&1))%max;
	}

}

//generate the  each value for <dup> tuples.
//dup=1,2,4,8,16,32
void generateSkewDuplicates(Record *R,  int rLen,Record *S, int sLen, int max, int dup, int seed)
{
	int a=0;
	int i=0;
	int minmin=0;
	int maxmax=2;
	unsigned int mask=(2<<15)-1;
	int seg=rLen/dup;
	srand(seed);
	for(i=0;i<seg;i++)
	{
		R[i].y=((((rand()& mask)<<15)+(rand()&1))+(rand()<<1)+(rand()&1))%max;
		if(i==0)
		{
			minmin=maxmax=R[i].y;
		}
		else
		{
			if(minmin>R[i].y) minmin=R[i].y;
			if(maxmax<R[i].y) maxmax=R[i].y;
		}
		R[i].x=i+1;
	}
	//copy the seg to all other segs.
	for(a=1;a<dup;a++)
	{
		for(i=0;i<seg;i++)
			R[a*seg+i].y=R[i].y;
	}
	const int offset=(1<<15)-1;
	for(i=0;i<sLen;i++)
	{
		S[i].x=i+1;
		S[i].y=((((rand()& offset)<<15)+(rand()&1))+(rand()<<1)+(rand()&1))%max;
	}
	//cout<<"min, "<<minmin<<", max, "<<maxmax<<", rand max, "<<max<<", dup, "<<dup<<endl;
#ifdef DEBUG_SAVEN
	printf("Be careful!!! DEBUGGING IS ENABLED\n");
	qsort(R,rLen,sizeof(Record),compare);
	qsort(S,sLen,sizeof(Record),compare);
#endif
}

void generateJoinSelectivity(Record *R, int rLen, Record *S, int sLen, int max, float joinSel,int seed)
{
	int i=0;
	const int offset=(1<<15)-1;
	srand(seed);
	for(i=0;i<rLen;i++)
	{
		R[i].y=((((rand()& offset)<<15)+(rand()&1))+(rand()<<1)+(rand()&1))%max;
		R[i].x=i+1;
	}
	for(i=0;i<sLen;i++)
	{
		S[i].x=-1;
		S[i].y=((((rand()& offset)<<15)+(rand()&1))+(rand()<<1)+(rand()&1))%max;
	}
	int locR=0;
	int locS=0;
	int retry=0;
	const int MAX_RET=1024;
	double deltaSel=(double)(rLen)/(double)max/1.25;
	joinSel-=(float)deltaSel;
	printf("%f,%f,",deltaSel,joinSel);
	if(joinSel<0)
	{
		joinSel=0-joinSel;
		int numMisses=(int)(joinSel*(float)sLen);
		for(i=0;i<numMisses;i++)
		{
			locR=((((rand()& offset)<<15)+(rand()&1))+(rand()<<1)+(rand()&1))%rLen;
			locS=((((rand()& offset)<<15)+(rand()&1))+(rand()<<1)+(rand()&1))%sLen;
			if(S[locS].x==-1)
			{
				S[locS].y=((((rand()& offset)<<15)+(rand()&1))+(rand()<<1)+(rand()&1))%max;
				S[locS].x=1;
				retry=0;
			}
			else
			{
				retry++;
				i--;
				if(retry>MAX_RET)
					break;
			}
		}
	}
	else
	{
		int numHits=(int)(joinSel*(float)sLen);
		for(i=0;i<numHits;i++)
		{
			locR=((((rand()& offset)<<15)+(rand()&1))+(rand()<<1)+(rand()&1))%rLen;
			locS=((((rand()& offset)<<15)+(rand()&1))+(rand()<<1)+(rand()&1))%sLen;
			if(S[locS].x==-1)
			{
				S[locS].y=R[locR].y;
				S[locS].x=1;
				retry=0;
			}
			else
			{
				retry++;
				i--;
				if(retry>MAX_RET)
					break;
			}
		}
	}
	for(i=0;i<sLen;i++)
	{
		S[i].x=i+1;
	}
	//for testing
#ifdef DEBUG_SAVEN
	printf("Be careful!!! DEBUGGING IS ENABLED\n");
	qsort(R,rLen,sizeof(Record),compare);
	qsort(S,sLen,sizeof(Record),compare);
#endif
}

void generateArray(int *R, int base, int step, int max, int rLen, int seed)
{
	int i=0;
	const int offset=(1<<15)-1;
	srand(seed);
	for(i=0;i<rLen;i++)
	{
		R[i*step+base]=((((rand()& offset)<<15)+(rand()&1))+(rand()<<1)+(rand()&1))%max;
	}
}

/*
 *	generate <rLen> sorted Record, in ascending order.
 */

void generateSort(Record *R, int maxmax, int rLen, int seed)
{
	int i=0;
	const int offset=(1<<15)-1;
	srand(seed);
	for(i=0;i<rLen;i++)
	{
		R[i].y=((((rand()& offset)<<15)+(rand()&1))+(rand()<<1)+(rand()&1))%maxmax;
		
	}
	qsort(R,rLen,sizeof(Record),compare);
	for(i=0;i<rLen;i++)
	R[i].x=i;

}

/************************************************************************/
/* This function generates <rLen> distinct tuples; distinct.
/************************************************************************/
/* (1) generate N^0.5 16-bit distinct numbers  (stored in array a);
   (2) generate another N^0.5 16-bit distinct numbers  (stored in array b);
   (3) the result array, x: x[i*N^0.5+j] =(a[i]<<16)��b[j]                 
/************************************************************************/
//step (1) and (2)
void generate16Bits(int *a, int max, int len, int seed)
{
	const int mask=(1<<16)-1;
	int i=0;
	int j=0;
	int temp=0;
	srand(seed);
	for(i=0;i<len;i++)
	{
		temp=(((rand()<<1)+(rand()&1))&mask)%max;
		for(j=0;j<i;j++)
			if(temp==a[j])
				break;
		if(j==i)
			a[i]=temp;
		else
			i--;	
	}
	//for(i=0;i<len;i++)
	//	printf("%d,",a[i]);
	//printf("\n");
	
}
void generateDistinct(Record *R, int max, int rLen, int seed)
{
	int i=0;
	int j=0;
	int curNum=0;
	int done=0;
	int nSquareRoot=(int)sqrt((double)rLen)+1;
	int *a=(int *)malloc(sizeof(int)*nSquareRoot);
	int *b=(int *)malloc(sizeof(int)*nSquareRoot);
	int maxSqrt=((int)sqrt((double)max)+1);
	generate16Bits(a,maxSqrt,nSquareRoot,seed);
	generate16Bits(b,maxSqrt,nSquareRoot,seed+1);
	for(i=0;i<nSquareRoot && !done;i++)
		for(j=0;j<nSquareRoot;j++)
		{
			R[curNum].y=(a[i]*maxSqrt)+b[j];
			R[curNum].x=curNum;
			curNum++;
			if(curNum==rLen)
			{
				done=1;
				break;
			}		
		}
	free(a);
	free(b);
}




void print(Record *R, int rLen)
{
	int i=0;
	printf("Random max=%d\n",RAND_MAX);
	for(i=0;i<rLen;i++)
	{
		printf("%d,%d\n",R[i].x, R[i].y);
	}
}

void generateSkew(Record *R, int max, int rLen, float oneRatio, int seed)
{
	int numOnes=(int)(((float)rLen)*oneRatio);
	int i=0;
	int onePos=0;
	const int offset=(1<<15)-1;
	srand(seed);	
	for(i=0;i<rLen;i++)
	{
		R[i].y=((((rand()& offset)<<15)+(rand()&1))+(rand()<<1)+(rand()&1))%max;
		R[i].x=i;
		if(R[i].y==1)
			numOnes--;
	}
	for(i=0;i<numOnes;i++)
	{
		onePos=((((rand()& offset)<<15)+(rand()&1))+(rand()<<1)+(rand()&1))%rLen;
		R[onePos].x=onePos;
		R[onePos].y=1;
	}	
	/*int numOnes=(int)((double)rLen*oneRatio);
	int i=0;
	for(i=0;i<numOnes;i++)
	{
		R[i].y=1;
		R[i].x=i;
	}
	const int offset=(1<<15)-1;
	srand(seed);
	for(;i<rLen;i++)
	{
		R[i].y=((((rand()& offset)<<15)+(rand()&1))+(rand()<<1)+(rand()&1))%max;
		R[i].x=i;
	}
	//randomize the array
	randomize(R, rLen, numOnes);*/
	//randomize the array
	//randomize(R, rLen, numOnes);
}


/*
*/
int computeMatrix(float3 *inputList, int n, int nonZero)
{
	char fileName[100];
	sprintf(fileName, "M%d.txt",n);
	FILE *src = fopen(fileName, "r");
	if(src!=NULL)
	{
		//load the data from the file.
		printf("loading data from file, %s, ", fileName);
		int rLen=0;
		int a, b;
		float c;
		 while (!feof(src)) 
		 {
			fscanf (src, "%d", &a);
			if(feof(src)) break;
			fscanf (src, "%d", &b);
			if(feof(src)) break;
			fscanf (src, "%f", &c);
			if(feof(src)) break;
			inputList[rLen].x=(float)a;
			inputList[rLen].y=(float)b;
			inputList[rLen].z=c;
			rLen++;
		}
		fclose(src);
		return rLen;
	}
	else
	{
		//fclose(src);
		float** A=(float **)malloc(sizeof(float *)*n);
		for(int i=0;i<n;i++)
			A[i]=(float *)malloc(sizeof(float)*n);
		printf("create %s", fileName);
		float *w=(float *)malloc(sizeof(float)*n);
		w[0]=0.1; w[n-1]=1.0;
		float q=(float)pow((float)10.0, (float)1.0/(float)n);
		int i=0;
		for(i=1;i<n-1;i++)
			w[i]=w[i-1]*q;
		float *x=(float*)malloc(sizeof(float)*n);
		int j=0,m=0;
		int tempIndex;
		for(j=0;j<n;j++)
				for(m=0;m<n;m++)
				A[j][m]=0;
		int tempValue=0;
		srand(0);
		for(i=0;i<n;i++)//the main loop
		{
			for(j=0;j<n;j++)
				x[j]=0;
			for(j=0;j<nonZero;j++)
			{
				tempIndex=rand()%n;
				tempValue=rand()%((1<<16)-1);
				x[tempIndex]=(float)tempValue/(float)((1<<16)-1);
			}
			x[i]=0.5;
			//compute xTx and add it to the A.
			for(j=0;j<n;j++)
				for(m=0;m<n;m++)
				{
					A[j][m]+=w[i]*x[j]*x[m];
				}
		}
		for(i=0;i<n;i++)
			A[i][i]+=(float)0.1;
		//count the number of zeros;
		FILE *src2 = fopen(fileName, "w");
		assert(src2);
		int numNonZeros=0;
		for(j=0;j<n;j++)
				for(m=0;m<n;m++)
				{
					if(A[j][m]!=0)
					{
						fprintf(src2, "%d\n",j);
						fprintf(src2, "%d\n",m);
						fprintf(src2, "%f\n",A[j][m]);
						inputList[numNonZeros].x=j;
						inputList[numNonZeros].y=m;
						inputList[numNonZeros].z=A[j][m];
						numNonZeros++;
					}
				}
		fclose(src2);
		printf("numNonZero, %d\n",numNonZeros);
		//write the matrix to a file.
		free(x);
		free(w);
		free(A);	
		return numNonZeros;
	}
	
	
}



/************************************************************************/
/* Timing
/************************************************************************/
/*static clock_t g_startTime;

void startTime()
{
	 g_startTime= clock();
}
double endTime(char *info)
{
	double cpuTime;
	clock_t end = clock();
	cpuTime= (end-g_startTime)/ (double)CLOCKS_PER_SEC;
	printf("%s, time, %.3f\n", info, cpuTime);
	return cpuTime;
}*/
StopWatchInterface* g_startTime;
void startTime()
{
	CUT_SAFE_CALL( sdkCreateTimer( &g_startTime));
    CUT_SAFE_CALL( sdkStartTimer( &g_startTime));
}
double endTime(char *info)
{
	cudaThreadSynchronize();
	CUT_SAFE_CALL( sdkStopTimer( &g_startTime));
	double result=(double)sdkGetTimerValue(&g_startTime);
	printf("***%s, time, %f, ms***\n", info, result);
    CUT_SAFE_CALL( sdkDeleteTimer( &g_startTime));
	return result;
}



void startTimer(StopWatchInterface **timer)
{
    CUT_SAFE_CALL( sdkCreateTimer( timer));
    CUT_SAFE_CALL( sdkStartTimer( timer));
}


double endTimer(char *info, StopWatchInterface **timer)
{
	cudaThreadSynchronize();
	CUT_SAFE_CALL( sdkStopTimer( timer));
	double result=sdkGetTimerValue(timer);
	printf("***%s costs, %f, ms***\n", info, result);
    CUT_SAFE_CALL( sdkDeleteTimer( timer));
	return result;
}
int log2(int value)
{
	int result=0;
	while(value>1)
	{
		value=value>>1;
		result++;
	}
	return result;
}

int log2Ceil(int value)
{
	int result=log2(value);
	if(value>(1<<result))
		result++;
	return result;
}


static clock_t g_startSumTime;
static double g_totalTime;

void startSumTime()
{
	 g_startSumTime= clock();
}

void endSumTime()
{
	cudaThreadSynchronize();
	double cpuTime;
	clock_t end = clock();
	cpuTime= (end-g_startSumTime)/ (double)CLOCKS_PER_SEC;
	g_totalTime+=cpuTime;
}
double printSumTime(char *info)
{
	double cpuTime;
	clock_t end = clock();
	cpuTime= (end-g_startSumTime)/ (double)CLOCKS_PER_SEC;
	g_totalTime+=cpuTime;
	printf("***%s costs, %f, ms***\n", info, g_totalTime);
	g_totalTime=0;
	return cpuTime;
}

#endif

