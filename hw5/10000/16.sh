#!/bin/sh
#PBS -N MABCB7_16_NODE_10000
#PBS -l nodes=16,walltime=01:00:00
#PBS -M mabcb7@mizzou.edu
#PBS -m abe
export OMP_NUM_THREADS=8
module load openmpi-x86_64
cd /cluster/students/7/mabcb7/HPC/hw5
mpirun -bynode -np 16 hw5 /cluster/content/hpc/distributed_data/ 10000
