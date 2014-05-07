#!/bin/sh
#PBS -N MABCB7_21_NODES
#PBS -l nodes=21,walltime=00:45:00
#PBS -M mabcb7@mizzou.edu
#PBS -m abe
export OMP_NUM_THREADS=8
module load openmpi-x86_64
cd /cluster/students/7/mabcb7/HPC/hw5
#mpirun directory_scanner2 .
#mpirun hw5 /cluster/students/7/mabcb7/HPC/hw5/test
mpirun -bynode -np 21 hw5 /cluster/content/hpc/distributed_data/
#mpirun hw5 /content/cs/hpc/data/sp14_1k.csv
#mpirun omptest
# Or you can run it this way also
#mpirun /cluster/students/7/mabcb7/a.out Michael
