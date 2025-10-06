#!/bin/bash

module load chemistry
module load schrodinger/2022-3

for dir in ./*/; do
    dir=${dir%*/}
    mkdir "${dir}"/lig

    rec_path="${dir}/${dir##*/}.maegz"
    if [ ! -f "$rec_path" ]; then
        echo "No receptor file found in $dir"
        continue
    fi
    for i in $(seq 0 15); do
        lig_path="${dir}/glide_maegz/lig_${i}.maegz"
        if [ ! -f "$lig_path" ]; then
            echo "No ligand file found in $dir"
            continue
        fi
        pv_path="${dir}/lig/lig_${i}.maegz"

        echo "$rec_path and $lig_path to be concatenated into $pv_path"
        "$SCHRODINGER"/utilities/structcat -i "$rec_path" "$lig_path" -o "$pv_path" &
    done
done

wait
