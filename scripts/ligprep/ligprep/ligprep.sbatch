#!/bin/bash
#SBATCH --job-name=ligprep
#SBATCH --output=ligprep_%A_%a.out
#SBATCH --error=ligprep_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zhaoyangli@stanford.edu
#SBATCH --partition=normal
#SBATCH --array=0-39
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=6G
#SBATCH --time=2-00:00:00

n=$SLURM_ARRAY_TASK_ID

# Load required modules
module load chemistry
module load schrodinger/2022-3
echo "Successfully loaded schrodinger/2022-3"

srun ligprep -f ligfilter.lff -ma 200 -epikx -ph 7.4 -pht 2.0 -g -bff 16 -NSTRUCTS 200 -RETRIES 2 -HOST "$HOSTNAME:$SLURM_CPUS_PER_TASK" -WAIT -TMPDIR "lig_${n}_tmp" -ismi "lig_${n}.smi" -osd "lig_${n}.sdf"
