#!/bin/bash

touch lig_test.log

for i in $(seq 0 39); do
    if test -s "lig_${i}.smi"; then
        if test -s "lig_${i}.sdf"; then
            if grep -Eq "All jobs have completed." "lig_${i}.log" && grep -Eq "[0-9]+ of [0-9]+ job\(s\) succeeded; 0 job\(s\) failed." "lig_${i}.log"; then
                continue
            else
                echo "lig_${i} has not been prepared because some of the subjobs failed!"
                echo -n "${i}," >>lig_test.log
            fi
        else
            echo "lig_${i} has not been prepared because the job died!"
            echo -n "${i}," >>lig_test.log
        fi
    fi
done
