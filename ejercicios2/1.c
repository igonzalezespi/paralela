#include <omp.h>
#include <stdio.h>

void tarea_uno() { printf("tarea_uno\n"); }
void tarea_dos() { printf("tarea_dos\n"); }

int main() {
  double timeIni, timeFin;

#pragma omp parallel sections
  {
#pragma omp section
    {
      timeIni = omp_get_wtime();
      printf("Ejecutando tarea 1\n");
      tarea_uno();
      timeFin = omp_get_wtime();
      printf("Tiempo total = %f segundos\n", timeFin - timeIni);
    }
#pragma omp section
    {
      timeIni = omp_get_wtime();
      printf("Ejecutando tarea 2\n");
      tarea_dos();
      timeFin = omp_get_wtime();
      printf("Tiempo total = %f segundos\n", timeFin - timeIni);
    }
  }

  return 0;
}
