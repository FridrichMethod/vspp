#!/bin/bash

mkdir lig

module load chemistry
module load schrodinger/2022-3
echo "Successfully loaded schrodinger/2022-3"

for i in $(seq 0 15); do
    "$SCHRODINGER"/utilities/glide_sort "glide_maegz/lig_$((2 * i)).maegz" "glide_maegz/lig_$((2 * i + 1)).maegz" -o "lig/lig_${i}.maegz" -use_dscore -maxsize 0 -norecep -best_by_title &
done

wait

rm -rf glide_maegz
