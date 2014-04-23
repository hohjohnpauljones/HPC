#!/bin/sh
#PBS -N MABCB7_5_NODE
#PBS -l nodes=5,walltime=00:45:00
#PBS -M mabcb7@mizzou.edu
#PBS -m abe
export OMP_NUM_THREADS=8
module load openmpi-x86_64
cd /cluster/students/7/mabcb7/HPC/hw5
#mpirun hw5 /cluster/students/7/mabcb7/HPC/hw5/test
mpirun -bynode -np 5 hw5 /cluster/content/hpc/dev_data 100
