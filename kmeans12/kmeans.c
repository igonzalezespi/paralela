#include "kmeans.h"
#define wtime omp_get_wtime

int randomOMP[256] =
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

void fail(const char *str) {
  fprintf(stderr, "%s", str);
  exit(-1);
}

float calc_distance(int dim, float *p1, float *p2) {
  float distance_sq_sum = 0.0f;
  for (int i = 0; i < dim; ++i) distance_sq_sum += sqr(p1[i] - p2[i]);
  return distance_sq_sum;
}

void calc_all_distances(int dim, int n, int k, float *X, float *centroid,
                        float *distance_output) {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < k; ++j)
      distance_output[i * k + j] =
          calc_distance(dim, &X[i * dim], &centroid[j * dim]);
}

float calc_total_distance(int dim, int n, int k, float *X, float *centroids,
                          int *cluster_assignment_index) {
  float tot_D = 0;
#pragma omp parallel for schedule(static) reduction(+ : tot_D)
  for (int i = 0; i < n; ++i) {
    int active_cluster = cluster_assignment_index[i];
    if (active_cluster != -1)
      tot_D +=
          calc_distance(dim, &X[i * dim], &centroids[active_cluster * dim]);
  }
  return tot_D;
}

void choose_all_clusters_from_distances(int dim, int n, int k,
                                        float *distance_array,
                                        int *cluster_assignment_index) {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; ++i) {
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

void calc_cluster_centroidsOMP(int dim, int n, int k, float *X,
                               int *cluster_assignment_index,
                               float *new_cluster_centroid) {
  int *cluster_member_count = (int *)malloc(k * sizeof(int));
  for (int i = 0; i < k; ++i) {
    cluster_member_count[i] = 0;
    for (int j = 0; j < dim; ++j) new_cluster_centroid[i * dim + j] = 0;
  }

#pragma omp parallel
  {
    int threadnum = omp_get_thread_num();
    int numthreads = omp_get_num_threads();
    int alto;
    int bajo;

    float *P_new_cluster_centroid = (float *)malloc(k * dim * sizeof(float));
    int *P_cluster_member_count = (int *)malloc(k * sizeof(float));

    for (int i = 0; i < k; i++) {
      P_cluster_member_count[i] = 0;
      for (int j = 0; j < dim; j++) {
        P_new_cluster_centroid[i * dim + j] = 0;
      }
    }

    bajo = n * threadnum / numthreads;
    alto = n * (threadnum + 1) / numthreads;
    for (int i = bajo; i < alto; i++) {
      int active_cluster = cluster_assignment_index[i];
      P_cluster_member_count[active_cluster]++;

      for (int j = 0; j < dim; j++) {
        P_new_cluster_centroid[active_cluster * dim + j] += X[i * dim + j];
      }
    }

#pragma omp critical
    {
      for (int i = 0; i < k; i++) {
        cluster_member_count[i] += P_cluster_member_count[i];
        for (int j = 0; j < dim; j++)
          new_cluster_centroid[i * dim + j] +=
              P_new_cluster_centroid[i * dim + j];
      }
    }
    free(P_cluster_member_count);
    free(P_new_cluster_centroid);
  }

  for (int i = 0; i < k; ++i)
    if (cluster_member_count[i] != 0)
      for (int j = 0; j < dim; j++)
        new_cluster_centroid[i * dim + j] /= cluster_member_count[i];
}

void get_cluster_member_count(int n, int k, int *cluster_assignment_index,
                              int *cluster_member_count) {
  for (int i = 0; i < k; i++) cluster_member_count[i] = 0;
  for (int i = 0; i < n; i++)
    cluster_member_count[cluster_assignment_index[i]]++;
}

void cluster_diag(int dim, int n, int k, float *X,
                  int *cluster_assignment_index, float *cluster_centroid) {
  int *cluster_member_count = (int *)malloc(k * sizeof(int));

  get_cluster_member_count(n, k, cluster_assignment_index,
                           cluster_member_count);

  printf("  Final clusters \n");
  for (int i = 0; i < k; ++i) {
    printf("\tcluster %d:  members: %8d, for the centroid (", i,
           cluster_member_count[i]);
    for (int j = 0; j < dim; j++) printf("%f, ", cluster_centroid[i * dim + j]);
    printf(")\n");
  }
}

void copy_assignment_array(int n, int *src, int *tgt) {
  for (int i = 0; i < n; i++) tgt[i] = src[i];
}

int assignment_change_count(int n, int a[], int b[]) {
  int change_count = 0;
  for (int i = 0; i < n; ++i)
    if (a[i] != b[i]) change_count++;
  return change_count;
}

void random_init_centroid(float *cluster_centro_id, float *dataSetMatrix,
                          int clusters, int rows, int columns) {
  for (int i = 0; i < clusters; ++i) {
    for (int j = 0; j < columns; ++j) {
      cluster_centro_id[i * columns + j] =
          dataSetMatrix[randomOMP[i] * columns + j];
    }
  }
}

void kmeans(int dim, float *X, int n, int k, float *cluster_centroid,
            int iterations, int *cluster_assignment_final, int mode) {
  omp_set_num_threads(mode);
  float *dist = (float *)malloc(sizeof(float) * n * k);
  int *cluster_assignment_cur = (int *)malloc(sizeof(int) * n);
  int *cluster_assignment_prev = (int *)malloc(sizeof(int) * n);
  float *point_move_score = (float *)malloc(sizeof(float) * n * k);

  if (!dist || !cluster_assignment_cur || !cluster_assignment_prev ||
      !point_move_score)
    fail("Error allocating dist arrays\n");

  random_init_centroid(cluster_centroid, X, k, n, dim);
  calc_all_distances(dim, n, k, X, cluster_centroid, dist);
  choose_all_clusters_from_distances(dim, n, k, dist, cluster_assignment_cur);
  copy_assignment_array(n, cluster_assignment_cur, cluster_assignment_prev);

  float prev_totD = calc_total_distance(dim, n, k, X, cluster_centroid,
                                        cluster_assignment_cur);
  int numVariations = 0;
  for (int batch = 0; (batch < iterations); ++batch) {
    calc_cluster_centroidsOMP(dim, n, k, X, cluster_assignment_cur,
                              cluster_centroid);
    float totD = calc_total_distance(dim, n, k, X, cluster_centroid,
                                     cluster_assignment_cur);
    if (totD >= prev_totD) {
      copy_assignment_array(n, cluster_assignment_prev, cluster_assignment_cur);
      random_init_centroid(cluster_centroid, X, k, n, dim);
    } else {
      copy_assignment_array(n, cluster_assignment_cur, cluster_assignment_prev);
      calc_all_distances(dim, n, k, X, cluster_centroid, dist);
      choose_all_clusters_from_distances(dim, n, k, dist,
                                         cluster_assignment_cur);
      prev_totD = totD;
    }
  }
  copy_assignment_array(n, cluster_assignment_cur, cluster_assignment_final);

  free(dist);
  free(cluster_assignment_cur);
  free(cluster_assignment_prev);
  free(point_move_score);
}
