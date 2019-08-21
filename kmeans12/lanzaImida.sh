#!/bin/bash

make clean; make


data=(`ls ./DatosIMIDA|sort -n`)
iter=( 1 )
clusters=( 2 )
threads=( 0 )


    #echo Running the classificaction for $input dataset
    
for i in ${iter[@]}
do 
    for c in ${clusters[@]}
    do
        echo Para $c clusters
        for input in ${data[@]}
        do
        #echo Para $input variables
            for proc in ${threads[@]} 
            do
                ./run -f DatosIMIDA/$input -i $i -c $c -m $proc
            done
        done
    done
done 

