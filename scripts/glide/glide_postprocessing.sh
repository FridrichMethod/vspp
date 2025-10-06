#!/bin/bash

mkdir ../glide_csv
mkdir ../glide_lib_maegz
mkdir ../glide_pv_maegz
mkdir ../glide_maegz
mkdir ../glide

module load chemistry
module load schrodinger/2022-3

filter=-6.5

file_maxnum=31

for i in $(seq 0 ${file_maxnum}); do
    if test -f "lig_${i}.sdf"; then
        mv "lig_${i}.sdf" ../glide
    fi
    if test -f "lig_${i}.maegz"; then
        mv "lig_${i}.maegz" ../glide
    fi

    if test -s "lig_${i}.csv"; then
        mv "lig_${i}.csv" ../glide_csv
        if test -f "lig_${i}_lib.maegz"; then
            "$SCHRODINGER"/utilities/glide_sort "lig_${i}_lib.maegz" -o "../glide_maegz/lig_${i}.maegz" -nosort -dscore_cut "${filter}" &
        fi
        if test -f "lig_${i}_pv.maegz"; then
            "$SCHRODINGER"/utilities/glide_sort "lig_${i}_pv.maegz" -o "../glide_maegz/lig_${i}.maegz" -nosort -dscore_cut "${filter}" &
        fi
        rm -rf "lig_${i}_tmp"
    fi
done

wait

for i in $(seq 0 ${file_maxnum}); do
    if test -f "lig_${i}_lib.maegz"; then
        mv "lig_${i}_lib.maegz" ../glide_lib_maegz
    fi
    if test -f "lig_${i}_pv.maegz"; then
        mv "lig_${i}_pv.maegz" ../glide_pv_maegz
    fi
done

cd ..
mv lig glide_log
mv glide lig
cd glide_log || exit
