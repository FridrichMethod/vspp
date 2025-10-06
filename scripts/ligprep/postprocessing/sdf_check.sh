#!/bin/bash

module load system
module load pcre

file=lig_13.sdf
lignum=6407

line_number=$(cat $file | grep -n '$$$$' | awk -F ':' '{print $1}' | sed -n "$((lignum - 1))p")

if [[ -n $line_number ]]; then
    awk -v start="$line_number" 'NR>start && NR<=start+1000' "$file" >"$(basename "$file" .sdf)-${line_number}.txt"
else
    echo "Ligand number not found"
fi
