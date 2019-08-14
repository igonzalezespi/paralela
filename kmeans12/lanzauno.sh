#!/bin/bash

make clean
make

for i in {1..4}
do
  ./run -f ./DataSet/datosMascados.txt -i 1000 -c 4
done
