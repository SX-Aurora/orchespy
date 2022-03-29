#include <stdio.h>
#include <assert.h>
#define __USE_GNU
#include <dlfcn.h>

// Set how many calls will cause an error.
#define VEO_READ_MEM_COUNT_ERROR 1
#define VEO_WRITE_MEM_COUNT_ERROR 0

int count_veo_read_mem;
int count_veo_write_mem;

int veo_read_mem(void * h, void * dst, int src, size_t size) {
    fprintf(stdout, "veo_read_mem(%x, %x, %x, %d)\n", h, dst, src, size);
    fprintf(stdout, "veo_read_mem call count = %d\n",count_veo_read_mem);
    if(count_veo_read_mem == VEO_READ_MEM_COUNT_ERROR){
    	return(-1);
    }
    count_veo_read_mem++;
    return(0);
}

int veo_write_mem(void * h, void * dst, int src, size_t size) {
    fprintf(stdout, "veo_write_mem(%x, %x, %x, %d)\n", h, dst, src, size);
    fprintf(stdout, "veo_write_mem call count = %d\n",count_veo_read_mem);
    if(count_veo_write_mem == VEO_WRITE_MEM_COUNT_ERROR){
    	return(-1);
    }
    count_veo_write_mem++;
    return(0);
}

