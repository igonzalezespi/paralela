#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

double wtime() {
  static int sec = -1;
  struct timeval tv;
  gettimeofday(&tv, NULL);
  if (sec < 0) sec = tv.tv_sec;
  return (tv.tv_sec - sec) + 1.0e-6 * tv.tv_usec;
}

__global__ void saxpy(int n, float a, float *x, float *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
  int N = 1 + 20;
  float *x, *y, *d_x, *d_y;
  int memsize = N * sizeof(float);
  x = (float *)malloc(memsize);
  y = (float *)malloc(memsize);

  cudaMalloc(&d_x, memsize);
  cudaMalloc(&d_y, memsize);

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaMemcpy(d_x, x, memsize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, memsize, cudaMemcpyHostToDevice);

  saxpy<<<(N + 255) / 256, 256>>>(N, 2.0f, d_x, d_y);

  cudaMemcpy(y, d_y, memsize, cudaMemcpyDeviceToHost);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++) {
    maxError = max(maxError, abs(y[i] - 4.0f));
  }

  printf("Max error: %f\n", maxError);

  return 0;
}