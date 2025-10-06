#!/bin/bash

n=30
k=0
for i in $(seq 1 2000); do
    if test -s "lig_${n}-${j}_in.maegz"; then
        j=$(printf "%04d" "${i}")
        if test -s "lig_${n}-${j}_raw.maegz"; then
            zcat "lig_${n}-${j}_raw.maegz" | gzip -c >>"lig_${n}_raw.maegz"
            if [ $k -eq 0 ]; then
                cat "lig_${n}-${j}.csv" >"lig_${n}_raw.csv"
            else
                tail -n +2 "lig_${n}-${j}.csv" >>"lig_${n}_raw.csv"
            fi
            k=1
        else
            zcat "lig_${n}-${j}_in.maegz" | gzip -c >>"lig_${n}_in.maegz"
        fi
    fi
done
