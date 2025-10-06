#!/bin/bash
#SBATCH --job-name=mmgbsa
#SBATCH --output=mmgbsa_%j.out
#SBATCH --error=mmgbsa_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zhaoyangli@stanford.edu
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=12G
#SBATCH --time=2-00:00:00

module load chemistry
module load schrodinger/2022-3
echo "Successfully loaded schrodinger/2022-3"

srun "$SCHRODINGER"/utilities/glide_sort mmgbsa_maegz/* -o "mmgbsa.maegz" -r "mmgbsa.txt" -use_prop r_psp_MMGBSA_dG_Bind -maxsize 0
