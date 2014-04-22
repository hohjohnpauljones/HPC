#!/bin/bash
echo Homework 5 with 5, 10, 15, 20, and 25 compute nodes
cd 100
echo 100 nearest neighbors
qsub 6.sh
#qsub 11.sh
qsub 16.sh
#qsub 21.sh
qsub 26.sh
cd ../1000
echo 1000 nearest neighbors
qsub 6.sh
#qsub 11.sh
qsub 16.sh
#qsub 21.sh
qsub 26.sh
#cd ../10000
#echo 10000 nearest neighbors
#qsub 6.sh
#qsub 11.sh
#qsub 16.sh
#qsub 21.sh
#qsub 26.sh
