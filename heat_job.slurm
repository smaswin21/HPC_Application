#!/bin/bash
#SBATCH --job-name=heat_parallel
#SBATCH --nodes=1                  # Request only 1 node
#SBATCH --ntasks-per-node=2        # Request 2 tasks per node
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=heat_parallel.out

module load gcc/9.3.0
module load openmpi/4.0.3

mpirun -np 2 ./heat_parallel
