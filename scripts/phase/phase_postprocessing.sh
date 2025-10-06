#!/bin/bash

mkdir ../phase_input
mkdir ../phase_maegz
mkdir ../phase_error

for i in $(seq 0 15); do
    if test -s "phase_${i}-hits.maegz"; then
        rm -rf "lig_${i}_tmp"
        mv "phase_${i}-hits.maegz" ../phase_maegz
    else
        echo "lig_${i} has no results!"
        mv "lig_${i}_tmp" ../phase_error
    fi
    mv "lig_${i}.maegz" ../phase_input
    mv ./*.phypo ../phase_input
done

cd ..
mv lig phase_log
mv phase_input lig
cd phase_log || exit
