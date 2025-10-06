#!/bin/bash

for i in $(seq 0 15); do
    if [[ ! -s lig_${i}-out.maegz ]]; then
        echo -n "${i},"
    else
        mv lig_"${i}".0????-out.maegz lig_"${i}"_tmp/
    fi
done

echo ""

# for i in $(seq 0 15); do if [[ ! -s lig_${i}-out.maegz ]]; then echo -n "${i},"; else mv lig_"${i}".0????-out.maegz lig_"${i}"_tmp/; fi; done
