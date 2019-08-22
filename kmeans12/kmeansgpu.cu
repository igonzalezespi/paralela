#include <curand.h>
#include <curand_kernel.h>
#include <sys/time.h>
#include "cuda.h"
#include "device_functions.h"
#include "kmeansgpu.h"

double cudaini, cudafin, cudatime = 0;
int H_randomCUDA[256] =
    {  // Datos pregenerados para que de el mismo resultado en OMP y CUDA
        219523, 10350,  46213,  232555, 170597, 60743,  170306, 154400, 181297,
        79121,  146366, 8043,   50882,  232179, 46147,  133302, 112968, 49690,
        182904, 78028,  95443,  100820, 121511, 118453, 40826,  232730, 162042,
        56607,  136199, 51455,  182481, 61914,  61806,  173994, 661,    177703,
        234737, 170968, 38295,  176926, 195389, 184661, 130269, 7163,   123032,
        121717, 85766,  236000, 116707, 29562,  20221,  212150, 75683,  141732,
        91496,  116509, 80654,  198838, 173116, 162153, 11185,  116489, 224068,
        72991,  235784, 170029, 195994, 231413, 47189,  234289, 114532, 187879,
        125142, 5693,   195042, 9067,   72710,  226108, 190367, 189418, 200971,
        155888, 162460, 37546,  58512,  199256, 99355,  139167, 158986, 33363,
        62212,  170172, 95152,  231580, 188463, 91828,  107802, 145350, 29434,
        154991, 85831,  89266,  103762, 210974, 40259,  4997,   165341, 112970,
        176405, 61900,  63280,  83568,  217789, 171040, 66414,  221601, 131189,
        165769, 66960,  51067,  144432, 74473,  166539, 477,    66945,  115895,
        37605,  120047, 206545, 12339,  220339, 237676, 101605, 30293,  154842,
        141865, 219698, 81075,  200135, 156996, 88276,  24307,  185864, 66957,
        140647, 13171,  49450,  32728,  178940, 116411, 29096,  84265,  136184,
        140935, 30042,  148429, 202130, 12947,  29369,  169567, 25287,  10600,
        113436, 72192,  40893,  29170,  214057, 205892, 110246, 120384, 123780,
        143822, 89991,  70536,  210779, 230639, 29007,  205529, 24259,  207948,
        28132,  237763, 237513, 164316, 139591, 212855, 73638,  102613, 225802,
        103007, 217481, 11981,  58907,  91809,  29474,  45100,  120979, 4423,
        11884,  176525, 124808, 80964,  81239,  214799, 96801,  237318, 151630,
        125808, 203740, 121190, 39948,  177172, 119845, 222761, 102381, 204736,
        196508, 121319, 13542,  183203, 169626, 231023, 140484, 228533, 29024,
        169958, 218933, 95303,  119682, 176118, 32721,  189790, 202382, 59260,
        110781, 5375,   2771,   23304,  131184, 151811, 144494, 116432, 89875,
        209639, 100086, 137556, 175268, 2786,   19767,  188810, 185989, 189393,
        126025, 32666,  124118, 155049};
int *D_randomCUDA;

double wtime() {
  static int sec = -1;
  struct timeval tv;
  gettimeofday(&tv, NULL);
  if (sec < 0) sec = tv.tv_sec;
  return (tv.tv_sec - sec) + 1.0e-6 * tv.tv_usec;
}

__device__ float calc_distanceCUDA(int dim, float *p1, float *p2) {
  float distance_sq_sum = 0.0f;
  for (int i = 0; i < dim; ++i) distance_sq_sum += sqr(p1[i] - p2[i]);

  return distance_sq_sum;
}

__global__ void calc_all_distancesCUDA(int dim, int n, int k, float *X,
                                       float *centroid,
                                       float *distance_output) {
  int i = threadIdx.y + blockIdx.y * blockDim.y;
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  // Asegurando que en la división (ceil) no hacemos más de la cuenta
  if (i < n && j < k)
    distance_output[i * blockDim.x + j] =
        calc_distanceCUDA(dim, &X[i * dim], &centroid[j * dim]);
}

__global__ void D_calc_total_distanceCUDA(int dim, float *D_X,
                                          float *D_centroids,
                                          int *D_cluster_assignment_index,
                                          float *D_tot) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int active_cluster = D_cluster_assignment_index[i];
  if (active_cluster != -1)
    atomicAdd(D_tot, calc_distanceCUDA(dim, &D_X[i * dim],
                                       &D_centroids[active_cluster * dim]));
}

float calc_total_distanceCUDA(int dim, int n, int k, float *D_X,
                              float *D_centroids,
                              int *D_cluster_assignment_index) {
  float H_tot = 0.0f;
  float *D_tot;
  cudaMalloc(&D_tot, sizeof(float));
  cudaMemcpy(
    D_tot,
     &H_tot, 
     sizeof(float), cudaMemcpyHostToDevice);
  // Parecido a blocksalldistances/threadsalldistances pero con una dimensión
  dim3 blocks(512);  // Dividimos "n" en bloques de 512
  dim3 threads((int)((n + 512 - 1) / 512));
  D_calc_total_distanceCUDA<<<blocks, threads>>>(
      dim,
       D_X, 
       D_centroids, 
       D_cluster_assignment_index, 
       D_tot);
  cudaMemcpy(&H_tot, &D_tot, sizeof(float), cudaMemcpyDeviceToHost);

  return H_tot;
};

void choose_all_clusters_from_distancesCUDA(int dim, int n, int k,
                                            float *distance_array,
                                            int *cluster_assignment_index) {
  for (int i = 0; i < n; ++i) {  // for each point
    int best_index = -1;
    float closest_distance = (__builtin_inff());

    for (int j = 0; j < k; j++) {
      float cur_distance = distance_array[i * k + j];
      if (cur_distance < closest_distance) {
        best_index = j;
        closest_distance = cur_distance;
      }
    }
    cluster_assignment_index[i] = best_index;
  }
}

__global__ void D_INIT_calc_cluster_centroidsCUDA(
    int dim, int n, int *D_cluster_member_count,
    float *D_new_cluster_centroid) {
  int i = threadIdx.y + blockIdx.y * blockDim.y;
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  // Aprovechamos para inicializar a 0, como en el código secuencial
  if (j == 0) D_cluster_member_count[i] = 0;
  if (i < n && j < dim) D_new_cluster_centroid[i * dim + j] = 0;
}
__global__ void D_SUM_calc_cluster_centroidsCUDA(
    int dim, int n, float *D_X, int *D_cluster_assignment_index,
    int *D_cluster_member_count, float *D_new_cluster_centroid) {
  int i = threadIdx.y + blockIdx.y * blockDim.y;
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int active_cluster = D_cluster_assignment_index[i];
  // Aprovechamos para sumar, solo en el "primer bucle"
  if (j == 0) D_cluster_member_count[active_cluster]++;

  if (i < n && j < dim)
    D_new_cluster_centroid[active_cluster * dim + j] += D_X[i * dim + j];
}
__global__ void D_DIV_calc_cluster_centroidsCUDA(
    int dim, int n, float *D_X, int *D_cluster_assignment_index,
    int *D_cluster_member_count, float *D_new_cluster_centroid) {
  // Cambiamos coordenadas i/j porque ahora el primer bucle es en la "k"
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (D_cluster_member_count[i] != 0)
    D_new_cluster_centroid[i * dim + j] /= D_cluster_member_count[i];
}

void calc_cluster_centroidsCUDA(int dim, int n, int k, float *D_X,
                                int *D_cluster_assignment_index,
                                float *H_new_cluster_centroid,
                                float *D_new_cluster_centroid, int ccsize) {
  // Lo mismo que blocksalldistances/threadsalldistances
  // pero ahora tomando "dim" en lugar de "k"
  dim3 blocks(1, 512);
  dim3 threads(dim, (int)((n + 512 - 1) / 512));

  int csize = k * sizeof(float);
  int *D_cluster_member_count;
  cudaMalloc(&D_cluster_member_count, csize);

  D_INIT_calc_cluster_centroidsCUDA<<<blocks, threads>>>(
      dim, n, D_cluster_member_count, D_new_cluster_centroid);
  // Tenemos que esperar a que termine de inicializar
  // Sumamos
  D_SUM_calc_cluster_centroidsCUDA<<<blocks, threads>>>(
      dim, n, D_X, D_cluster_assignment_index, D_cluster_member_count,
      D_new_cluster_centroid);
  // Tenemos que esperar a que termine de sumar
  // Dividimos
  D_DIV_calc_cluster_centroidsCUDA<<<blocks, threads>>>(
      dim, n, D_X, D_cluster_assignment_index, D_cluster_member_count,
      D_new_cluster_centroid);
  cudaMemcpy(H_new_cluster_centroid, D_new_cluster_centroid, ccsize,
             cudaMemcpyDeviceToHost);
}

void get_cluster_member_countCUDA(int n, int k, int *cluster_assignment_index,
                                  int *cluster_member_count) {
  for (int i = 0; i < k; i++) cluster_member_count[i] = 0;
  for (int i = 0; i < n; i++)
    cluster_member_count[cluster_assignment_index[i]]++;
}

void cluster_diagCUDA(int dim, int n, int k, float *X,
                      int *cluster_assignment_index, float *cluster_centroid) {
  int *cluster_member_count = (int *)malloc(k * sizeof(int));

  get_cluster_member_countCUDA(n, k, cluster_assignment_index,
                               cluster_member_count);

  printf("  Final clusters \n");
  for (int i = 0; i < k; ++i) {
    printf("\tcluster %d:  members: %8d, for the centroid (", i,
           cluster_member_count[i]);
    for (int j = 0; j < dim; j++) printf("%f, ", cluster_centroid[i * dim + j]);
    printf(")\n");
  }
}

void copy_assignment_arrayCUDA(int n, int *src, int *tgt) {
  for (int i = 0; i < n; i++) tgt[i] = src[i];
}

int assignment_change_countCUDA(int n, int a[], int b[]) {
  int change_count = 0;
  for (int i = 0; i < n; ++i)
    if (a[i] != b[i]) change_count++;
  return change_count;
}

__global__ void random_init_centroidCUDA(float *cluster_centro_id,
                                         float *dataSetMatrix, int clusters,
                                         int rows, int columns,
                                         int *D_randomCUDA) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = 0; i < columns; i++) {
    cluster_centro_id[index * columns + i] =
        dataSetMatrix[D_randomCUDA[index] * columns + i];
  }
}

extern "C" int kmeansCUDA(int dim, float *H_X, float n, int k,
                          float *H_cluster_centroid, int iterations,
                          int *H_cluster_assignment_final) {
  cudaDeviceReset();
  float prev_totD;
  int numVariations;

  /*** RANDOM ***/
  cudaMalloc(&D_randomCUDA, 256 * sizeof(int));
  cudaMemcpy(D_randomCUDA, H_randomCUDA, 256 * sizeof(int),
             cudaMemcpyHostToDevice);
  /*** FIN RANDOM***/

  // 0.000030 en las demás configuraciones, esta da 0.000028
  // Lo hacemos así para aprovechar, teóricamente, las
  // capacidades de todos los multiproc.
  dim3 blocksrandom(k);
  dim3 threadsrandom(1);

  // Tenemos probablemente muchos datos (n) por lo tanto, dividimos entre 512
  // para poder tener un thread para cada dato.
  // En caso de probar con más datos quizás tendríamos que ampliar a 1024.
  // Si aun así tenemos demasiados datos tendríamos que hacer un bucle en la
  // llamada al kernel repartiendo entre el máximo.
  dim3 blocksalldistances(1, 512);
  dim3 threadsalldistances(k, (int)((n + 512 - 1) / 512));

  // cudaini = wtime();
  // cudafin = wtime();
  // cudatime += cudafin - cudaini;

  int nsize = sizeof(int) * n;
  int nksize = sizeof(float) * n * k;
  int ccsize = sizeof(float) * dim * k;
  int casize = sizeof(int) * n;
  int xsize = sizeof(float) * dim * n;

  float *D_cluster_centroid;
  int *D_cluster_assignment_final;
  float *D_X;

  float *H_dist = (float *)malloc(nksize), *D_dist;
  int *H_cluster_assignment_cur = (int *)malloc(nsize),
      *D_cluster_assignment_cur;
  int *H_cluster_assignment_prev = (int *)malloc(nsize);
  float *H_point_move_score = (float *)malloc(nksize);

  cudaMalloc(&D_cluster_centroid, ccsize);
  cudaMalloc(&D_cluster_assignment_final, casize);
  cudaMalloc(&D_X, xsize);

  cudaMalloc(&D_dist, nksize);
  cudaMalloc(&D_cluster_assignment_cur, nsize);

  cudaMemcpy(D_cluster_centroid, H_cluster_centroid, ccsize,
             cudaMemcpyHostToDevice);
  cudaMemcpy(D_X, H_X, xsize, cudaMemcpyHostToDevice);

  cudaMemcpy(D_cluster_assignment_cur, H_cluster_assignment_cur, nsize,
             cudaMemcpyHostToDevice);
  // FIN DECLARACIÓN/RESERVA/COPIA

  // RANDOMS
  random_init_centroidCUDA<<<blocksrandom, threadsrandom>>>(
      D_cluster_centroid, D_X, k, n, dim, D_randomCUDA);
  cudaMemcpy(H_cluster_centroid, D_cluster_centroid, ccsize,
             cudaMemcpyDeviceToHost);

  // DISTANCIAS
  calc_all_distancesCUDA<<<blocksalldistances, threadsalldistances>>>(
      dim, n, k, D_X, D_cluster_centroid, D_dist);
  cudaMemcpy(H_dist, D_dist, nksize, cudaMemcpyDeviceToHost);

  // No merece la pena paralelizarla en CUDA.
  choose_all_clusters_from_distancesCUDA(dim, n, k, H_dist,
                                         H_cluster_assignment_cur);

  // No merece la pena paralelizarla en CUDA.
  copy_assignment_arrayCUDA(n, H_cluster_assignment_cur,
                            H_cluster_assignment_prev);
  prev_totD = calc_total_distanceCUDA(dim, n, k, H_X, H_cluster_centroid,
                                      H_cluster_assignment_cur);

  numVariations = 0;
  for (int batch = 0; (batch < iterations); ++batch) {
    cudaini = wtime();
    calc_cluster_centroidsCUDA(dim, n, k, D_X, D_cluster_assignment_cur,
                               H_cluster_centroid, D_cluster_centroid, ccsize);
    cudafin = wtime();
    cudatime += cudafin - cudaini;
    float totD = calc_total_distanceCUDA(dim, n, k, H_X, H_cluster_centroid,
                                         H_cluster_assignment_cur);
    if (totD >= prev_totD) {
      copy_assignment_arrayCUDA(n, H_cluster_assignment_prev,
                                H_cluster_assignment_cur);
      random_init_centroidCUDA<<<blocksrandom, threadsrandom>>>(
          D_cluster_centroid, D_X, k, n, dim, D_randomCUDA);
      cudaMemcpy(H_cluster_centroid, D_cluster_centroid, ccsize,
                 cudaMemcpyDeviceToHost);
    } else {
      copy_assignment_arrayCUDA(n, H_cluster_assignment_cur,
                                H_cluster_assignment_prev);
      // DISTANCIAS
      calc_all_distancesCUDA<<<blocksalldistances, threadsalldistances>>>(
          dim, n, k, D_X, D_cluster_centroid, D_dist);
      cudaMemcpy(H_dist, D_dist, nksize, cudaMemcpyDeviceToHost);
      choose_all_clusters_from_distancesCUDA(dim, n, k, H_dist,
                                             H_cluster_assignment_cur);
      prev_totD = totD;
    }
  }
  copy_assignment_arrayCUDA(n, H_cluster_assignment_cur,
                            H_cluster_assignment_final);

  free(H_dist);
  free(H_cluster_assignment_cur);
  free(H_cluster_assignment_prev);
  free(H_point_move_score);

  cudaFree(D_cluster_centroid);
  cudaFree(D_cluster_assignment_final);
  cudaFree(D_X);

  cudaFree(D_dist);
  cudaFree(D_cluster_assignment_cur);
}
