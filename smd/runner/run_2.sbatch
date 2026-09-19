#!/bin/bash
#SBATCH -J trdev009
#SBATCH -o Logs/ray_%j.out
#SBATCH -e Logs/ray_%j.err
#SBATCH -N 1
#SBATCH -c 112
#SBATCH -t 0-12:00
#SBATCH -p test
#SBATCH --mem=184G

module load python/3.10.13-fasrc01
mamba activate Tianlei_startup
# be sure that both -c terms agree
srun -c 112 python -u ray_SMD.py
