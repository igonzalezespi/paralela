#!/bin/bash

make clean >/dev/null
make >/dev/null

# data=(`ls ./DatosIMIDA|sort -n`)
data=($(ls ./DatosIMIDA/2variables.txt | sort -n))
iter=(1000)
clusters=(2 3 4 8 16 32 64 128 256)
threads=(0 1 32)

echo Running the classificaction for $input dataset

for i in ${iter[@]}; do
  for c in ${clusters[@]}; do
    echo Para $c clusters
    for input in ${data[@]}; do
      for proc in ${threads[@]}; do
        echo $proc threads
        ./run -f $input -i $i -c $c -m $proc
      done
    done
  done
done
