#!/bin/bash

make clean >/dev/null
make >/dev/null

./run -f ./DataSet/datosMascados.txt -i 1000 -c 4
