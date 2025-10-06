#!/bin/bash

lig=$(basename "$PWD")
i=1
j=0
dir=../../..

while [[ -s "in_${lig}-${i}.smi" ]]; do
    if [[ -s "${lig}-${i}.sdf" ]]; then
        if [[ $(tail -1 "${lig}-${i}.sdf") == '$$$$' ]]; then
            cat "${lig}-${i}.sdf" >>"${dir}/${lig}.sdf"
            ((j++))
        else
            echo "${lig}-${i} failed beacuse of incomplete sdf file"
            cp "in_${lig}-${i}.smi" "${dir}/in_${lig}-${i}.smi"
        fi
    else
        if [[ -s "${lig}-${i}-dropped.smi" ]]; then
            ((j++))
        else
            echo "${lig}-${i} failed because of no sdf file"
            cp "in_${lig}-${i}.smi" "${dir}/in_${lig}-${i}.smi"
        fi
    fi
    if [[ -s "${lig}-${i}-dropped.smi" ]]; then
        cat "${lig}-${i}-dropped.smi" >>"${dir}/${lig}-dropped.smi"
    fi
    ((i++))
done
((i--))

echo "${j}/${i} subjobs in ${lig} has been processed successfully"
