#include <helper_timer.h>
#  define CUT_SAFE_CALL( call)                                               \
    if( 1 != call) {                                                   \
        fprintf(stderr, "Cut error in file '%s' in line %i.\n",              \
                __FILE__, __LINE__);                                         \
        exit(EXIT_FAILURE);                                                  \
    }

typedef int2 Record;
void seed_generator(unsigned int);
int create_relation_pk(Record *, int);
void startTimer(StopWatchInterface **);
double endTimer(char *info, StopWatchInterface **);
int compare (const void * a, const void * b);
void generateRand(Record *R, int maxmax, int rLen, int seed);
void generateRand1(Record *R, int maxmax, int rLen, int seed);
void startTime();
double endTime(char *info);
int log2Ceil(int value);
void startSumTime();
void endSumTime();
double printSumTime(char *info);
int log2(int value);
void generateSort(Record *R, int maxmax, int rLen, int seed);
void generateRandInt(int *R, int max, int rLen, int seed);
void generateSkew(Record *R, int max, int rLen, float oneRatio, int seed);
void generateJoinSelectivity(Record *R, int rLen, Record *S, int sLen, int max, float joinSel,int seed);
