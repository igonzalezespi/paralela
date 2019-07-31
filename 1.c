#include <stdio.h>
#include <omp.h>

int main() {
	int num_threads;

	printf("Threads: ");
	scanf("%d", &num_threads);
	omp_set_num_threads(num_threads);
	#pragma omp parallel
	{
		printf("ID: %d\n", omp_get_thread_num());
	}

	return 0;
}
