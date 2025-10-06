#!/bin/bash

mkdir ../lig_input
mkdir ../lig_prepared
mkdir ../lig_dropped
mkdir ../lig_error
mkdir ../lig_failed

for i in $(seq 0 39); do
    if test -s "lig_${i}.smi"; then
        mv "lig_${i}.smi" ../lig_input
        if test -s "lig_${i}.sdf"; then
            rm -rf "lig_${i}_tmp"
            mv "lig_${i}.sdf" ../lig_prepared
            if test -s "lig_${i}-dropped.smi"; then
                echo "lig_${i} has dropped ligands!"
                mv "lig_${i}-dropped.smi" ../lig_dropped
            fi
            for file in "in_lig_${i}"-*.smi; do
                if [[ -s "$file" ]]; then
                    echo "lig_${i} has failed ligands!"
                    cat "in_lig_${i}"-*.smi >>"in_lig_${i}.smi"
                    mv "in_lig_${i}.smi" ../lig_failed
                    break
                fi
            done
        else
            echo "lig_${i} has not been prepared!"
            mv "lig_${i}_tmp" ../lig_error
        fi
    fi
done

cd ..
mv lig lig_log
mv lig_input lig
cd lig_log || exit
