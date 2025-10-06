#!/bin/bash

mkdir lig_separated
filename=lig.smi
temp_file=lig_separated/temp_file.smi
cp $filename $temp_file
lines_per_file=100000

i=0
sed -i '1d' $temp_file
while true; do
    echo "smiles title" >"lig_separated/lig_${i}.smi"
    head -n $lines_per_file $temp_file >>"lig_separated/lig_${i}.smi"
    sed -i -e "1,${lines_per_file}d" $temp_file
    echo "lig_${i}.smi created"
    if [[ ! -s $temp_file ]]; then
        rm $temp_file
        ((i++))
        echo "${i} files created"
        break
    fi
    ((i++))
done
