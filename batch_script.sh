#!/bin/bash
#SBATCH -A loni_deepvote2
#SBATCH -p gpu
#SBATCH -t 72:00:00
#SBATCH -N 1
#SBATCH -n 48
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH --mail-user lmatone@tulane.edu

module load conda
cd $SLURM_SUBMIT_DIR
source activate testing

./start dark_knight.txt "batman"
echo "./start dark_knight.txt “batman"

exit
