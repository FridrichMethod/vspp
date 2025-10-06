#!/bin/bash
#SBATCH --job-name=pose_extract
#SBATCH --output=pose_extract_%j.out
#SBATCH --error=pose_extract_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zhaoyangli@stanford.edu
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4G
#SBATCH --time=1-00:00:00

module load chemistry
module load schrodinger/2022-3
echo "Successfully loaded schrodinger/2022-3"

srun "$SCHRODINGER"/utilities/maesubset mmgbsa.maegz -title_list_file "title.txt" -o "manual_selection.maegz"
