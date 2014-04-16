#!/bin/sh
#PBS -N MABCB7_HW5_TEST
#PBS -l nodes=11,walltime=01:00:00
#PBS -M mabcb7@mizzou.edu
#PBS -m abe
module load openmpi-x86_64
cd /cluster/students/7/mabcb7/HPC/hw5
#mpirun directory_scanner2 .
#mpirun hw5 /cluster/students/7/mabcb7/HPC/hw5/test
mpirun hw5 /cluster/content/hpc/distributed_data/
#mpirun hw5 /content/cs/hpc/data/sp14_1k.csv
# Or you can run it this way also
#mpirun /cluster/students/7/mabcb7/a.out Michael
