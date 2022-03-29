#include <stdio.h>
#include <assert.h>
#define __USE_GNU
#include <dlfcn.h>

// Set how many calls will cause an error
#define CUDA_MEMCPY_COUNT_ERROR 0

int count_cudaMemcpy;

int cudaMemcpy(void * dst, void * buf, size_t size, int type) {
    fprintf(stdout, "cudaMemcpy(%x, %x, %d, %d)\n", dst, buf, size, type);
    fprintf(stdout, "cudaMemcpy call count =  %d\n",count_cudaMemcpy);
    if(count_cudaMemcpy == CUDA_MEMCPY_COUNT_ERROR){
    	return(-1);
    }
    count_cudaMemcpy++;
    return(0);
}
