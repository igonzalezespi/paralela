#include <omp.h>
#include <stdio.h>

int main() {
  int n = 5;

  printf("n: %d\n", n);

#pragma omp parallel for private(n)
  for (int i = 0; i < 10; i++) {
    n = n ? n + 1 : 0;
    printf("n: %d\n", n);
  }

  printf("n: %d\n", n);

  return 0;
}
