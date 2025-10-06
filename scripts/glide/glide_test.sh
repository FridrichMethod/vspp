#!/bin/bash

touch lig_test.log

for i in $(seq 0 20); do
    if test -s "lig_${i}.sdf" || test -s "lig_${i}.maegz"; then
        if test -s "lig_${i}.csv"; then
            if grep -Eq "All jobs have completed." "lig_${i}.log" && grep -Eq "[0-9]+ of [0-9]+ job\(s\) succeeded; 0 job\(s\) failed." "lig_${i}.log"; then
                continue
            else
                echo "lig_${i} has not been docked because some of the subjobs failed!"
                echo -n "${i}," >>lig_test.log
            fi
        else
            echo "lig_${i} has not been docked because the job died!"
            echo -n "${i}," >>lig_test.log
        fi
    fi
done
