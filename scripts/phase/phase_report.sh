#!/bin/bash

module load chemistry
module load schrodinger/2022-3
echo "Successfully loaded schrodinger/2022-3"

"$SCHRODINGER"/utilities/glide_sort phase_maegz/* -o "phase.maegz" -r "phase.txt" -use_prop_d r_phase_PhaseScreenScore -maxsize 0 -best_by_title
