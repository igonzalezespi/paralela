#include <omp.h>
#include <stdio.h>
void threads_por_nivel(int nivel) {
  printf("Nivel %d: Número de hilos en el nivel %d - %d\n", nivel, nivel,
         omp_get_num_threads());
}

int main() {
  omp_set_nested(1);
  omp_set_num_threads(2);
#pragma omp parallel
  {
    threads_por_nivel(1);
    omp_set_num_threads(2);
#pragma omp parallel
    {
      threads_por_nivel(2);
      omp_set_num_threads(2);
#pragma omp parallel
      { threads_por_nivel(3); }
    }
  }
  return (0);
}