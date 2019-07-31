struct timeval start, end;

gettimeofday(&start, NULL);

// benchmark code

gettimeofday(&end, NULL);
delta = ((end.tv_sec - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
