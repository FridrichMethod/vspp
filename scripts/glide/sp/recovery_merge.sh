#!/bin/bash

n=30

if test -s "lig_${n}_in_lib.maegz"; then
    "$SCHRODINGER"/utilities/glide_sort "lig_${n}_lib.maegz" "lig_${n}_raw.maegz" "lig_${n}_in_lib.maegz" -o "lig_${n}_lib_merged.maegz" -use_dscore -maxsize 0
    rm "lig_${n}_lib.maegz"
    mv "lig_${n}_lib_merged.maegz" "lig_${n}_lib.maegz"
    tail -n +2 "lig_${n}_raw.csv" "lig_${n}_in.csv" >>"lig_${n}.csv"
fi

rm -rf "lig_${n}_in_tmp"
