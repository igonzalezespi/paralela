#!/bin/bash

make clean; make


data=(`ls ./DataSet/datos*`)
iter=( 250 500 1000 )
clusters=( 2 3 4 )


for input in ${data[@]}
do
    echo Running the classificaction for $input dataset
    
    for i in ${iter[@]}
    do 
        for c in ${clusters[@]}
        do
            echo fichero:$input iterations:$i clusters $c

           ./run -f DataSetAlu/$input -i $i -c $c
        done
    done
done 

