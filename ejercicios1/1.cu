#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define N 8
#define THREADS 2

double wtime() {
  static int sec = -1;
  struct timeval tv;
  gettimeofday(&tv, NULL);
  if (sec < 0) sec = tv.tv_sec;
  return (tv.tv_sec - sec) + 1.0e-6 * tv.tv_usec;
}

__global__ void jacobi(float *a, float *b) {
  int i = blockIdx.x * blockDim.x * N + threadIdx.x + 1;
  if (i < N - 1) b[i] = 0.33333f * (a[i - 1] + a[i] + a[i + 1]);
}

int main() {
  float *h_a, *h_b;
  float *d_a, *d_b;

  h_a = (float *)malloc(N * sizeof(float));
  h_b = (float *)malloc(N * sizeof(float));

  cudaMalloc(&d_a, N * sizeof(float));
  cudaMalloc(&d_b, N * sizeof(float));

  for (int i = 1; i < N; i++) {
    h_a[i] = h_b[i] = 70.0f;
  }
  h_a[0] = h_a[N - 1] = h_b[0] = h_b[N - 1] = 150.0f;

  cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);

  dim3 block((N - 2) / THREADS);
  dim3 thread(THREADS);

  for (int i = 0; i < THREADS; i++) {
    jacobi<<<block, thread>>>(d_a, d_b);
    cudaThreadSynchronize();
    float *aux = d_a;
    d_a = d_b;
    d_b = aux;
  }

  cudaMemcpy(h_a, d_a, N * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; i++) printf("%2.0f, ", h_a[i]);
  printf("\n");

  // Liberamos memoria
  free(h_a);
  free(h_b);

  cudaFree(d_a);
  cudaFree(d_b);

  return 0;
}