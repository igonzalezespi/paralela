#include <omp.h>
#include <stdio.h>

int main() {
#pragma omp parallel
  {
#pragma omp master
    { printf("master  %d\n", omp_get_thread_num()); }
    printf("%d\n", omp_get_thread_num());
  }

  return 0;
}
