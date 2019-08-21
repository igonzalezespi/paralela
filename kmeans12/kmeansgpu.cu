#include <curand.h>
#include <curand_kernel.h>
#include "cuda.h"
#include "device_functions.h"
#include "kmeansgpu.h"

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    const char *error = cudaGetErrorString(code);
    fprintf(stderr, "GPUassert: %s %s %d\n", error, file, line);
    if (abort) exit(code);
  }
}

__global__ void random_init_centroidCUDA(float *cluster_centro_id,
                                         float *dataSetMatrix, int clusters,
                                         int rows, int columns) {
  int tx = threadIdx.x;
  int pos = tx * columns;
  int random = 0;
  for (int i = 0; i < columns; i++) {
    cluster_centro_id[pos + i] = dataSetMatrix[random + i];
  }
}

extern "C" int kmeansCUDA(int dim, float *H_X, float n, int k,
                          float *H_cluster_centroid, int iterations,
                          int *H_cluster_assignment_final) {
  int block = ceil((n / TH) / OPERTH);
  float *D_cluster_centroid;
  printf("\n\nhola\n\n");
  return 0;
}
