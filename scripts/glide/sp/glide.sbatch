#!/bin/bash
#SBATCH --job-name=glide
#SBATCH --output=glide_%A_%a.out
#SBATCH --error=glide_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zhaoyangli@stanford.edu
#SBATCH --partition=normal
#SBATCH --array=0-31%5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=6G
#SBATCH --time=2-00:00:00

n=$SLURM_ARRAY_TASK_ID

module load chemistry
module load schrodinger/2022-3
echo "Successfully loaded schrodinger/2022-3"

cat>"lig_${n}.in"<<EOF
EXPANDED_SAMPLING   True
FORCEFIELD   OPLS4
GRIDFILE   "grid.zip"
HBOND_DONOR_HALO   True
INCLUDE_INPUT_RINGS   True
LIGANDFILE   "lig_${n}.maegz"
MAXATOMS   200
MAXROTBONDS   40
POSE_OUTTYPE   ligandlib
POSTDOCKSTRAIN   True
PRECISION   SP
EOF
echo "Successfully created input file"

srun glide "lig_${n}.in" -adjust -max_retries 4 -HOST "$HOSTNAME:$SLURM_CPUS_PER_TASK" -WAIT -TMPDIR "lig_${n}_tmp" -NSTRUCTS 40 -RESTART
