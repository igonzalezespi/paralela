#!/bin/bash

make clean >/dev/null
make >/dev/null

for i in {1..4}
do
  ./run -f ./DataSet/datosMascados.txt -i 20000 -c 8
done