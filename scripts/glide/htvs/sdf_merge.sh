#!/bin/bash

mkdir lig
gap=2

i=0
while [[ -s "lig_prepared/lig_$((i * gap)).sdf" ]]; do
    for j in $(seq 0 $((gap - 1))); do
        k=$((i * gap + j))
        if [[ -s "lig_prepared/lig_${k}.sdf" ]]; then
            cat "lig_prepared/lig_${k}.sdf" >>"lig/lig_${i}.sdf"
        fi
    done
    echo "Successfully merged lig_${i}"
    ((i++))
done
