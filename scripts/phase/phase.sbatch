#!/bin/bash
#SBATCH --job-name=phase
#SBATCH --output=phase_%A_%a.out
#SBATCH --error=phase_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zhaoyangli@stanford.edu
#SBATCH --partition=normal
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4G
#SBATCH --time=2-00:00:00

n=$SLURM_ARRAY_TASK_ID

module load chemistry
module load schrodinger/2022-3
echo "Successfully loaded schrodinger/2022-3"

srun phase_screen "lig_${n}.maegz" "lig.phypo" "phase_${n}" -flex -sample thorough -max 1000 -force_field S-OPLS -append -match 3 -ex -atypes -keep 999999999 -report 1 -HOST "$HOSTNAME:$SLURM_CPUS_PER_TASK" -WAIT -TMPDIR "lig_${n}_tmp" -NJOBS 60
