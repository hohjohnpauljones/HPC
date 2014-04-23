#!/bin/sh
#PBS -N MABCB7_25_NODE_100
#PBS -l nodes=25,walltime=01:00:00
#PBS -M mabcb7@mizzou.edu
#PBS -m abe
export OMP_NUM_THREADS=8
module load openmpi-x86_64
cd /cluster/students/7/mabcb7/HPC/hw5
mpirun -bynode -np 25 hw5 /cluster/content/hpc/distributed_data/ 100
