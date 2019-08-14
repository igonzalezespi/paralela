//************************** kmeansgpu.cu ***************************
//*******************Developed by José M. Cecilia*******************
//************************* October 2018************************


#include "cuda.h"
#include "kmeansgpu.h"
#include <curand.h>
#include "device_functions.h"
#include <curand_kernel.h>


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		const char * error = cudaGetErrorString(code);
		fprintf(stderr, "GPUassert: %s %s %d\n", error, file, line);
		if (abort) exit(code);
	}
}

/*__global__ void setup_kernel(curandState *state, unsigned long seed) {
	
    int index = threadIdx.x;
	curand_init(seed, index, 0, &state[index]);
    __syncthreads();
}*/

__global__ void random_init_centroidCUDA(float * cluster_centro_id, float * dataSetMatrix, int clusters, int rows, int columns) {
	
    int tx = threadIdx.x;
    int pos=tx*columns;

//    int random = ceil(curand_uniform(&D_state[tx])*rows);
    int random =0;
    for (int i=0; i<columns; i++){
    	cluster_centro_id[pos+i] = dataSetMatrix[random+i];
        //printf ("El random es %f para el thread %d\n", cluster_centro_id[pos+i], tx);
    }   
}



extern "C" int kmeansCUDA(int  dim, float *H_X, float n, int k, float *H_cluster_centroid, int iterations, int *H_cluster_assignment_final) {



    cudaDeviceReset();
    return 0;


}






















