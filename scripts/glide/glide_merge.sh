#!/bin/bash

mkdir glide_merge

module load chemistry
module load schrodinger/2022-3

for i in $(seq 0 15); do
    files_to_merge=()

    # Loop over all directories in the current directory
    for dir in ./*/; do
        dir=${dir%*/}
        file_path="${dir}/glide_maegz/lig_${i}.maegz"
        if [[ -f "$file_path" ]]; then
            files_to_merge+=("$file_path")
        fi
    done

    # Check if there are any files to merge
    if [[ ${#files_to_merge[@]} -eq 0 ]]; then
        echo "No files to merge."
        exit 1
    fi

    # Echo all the files to be merged
    echo "Files to be merged:"
    printf "%s\n" "${files_to_merge[@]}"

    # Call the glide_merge utility with the file paths
    "$SCHRODINGER"/utilities/glide_merge "${files_to_merge[@]}" -o "glide_merge/lig_${i}.maegz" -p 1 -reset_lignum &
done

wait
