#!/bin/bash
echo Homework 5 with 5, 10, 15, 20, and 25 compute nodes
cd 100
echo 100 nearest neighbors
qsub 5.sh
qsub 10.sh
qsub 15.sh
qsub 20.sh
qsub 25.sh
cd ../1000
echo 1000 nearest neighbors
qsub 5.sh
qsub 10.sh
qsub 15.sh
qsub 20.sh
qsub 25.sh
#cd ../10000
#echo 10000 nearest neighbors
#qsub 5.sh
#qsub 10.sh
#qsub 15.sh
#qsub 20.sh
#qsub 25.sh
cd ../
