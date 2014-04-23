#!/bin/sh
DIRS="100 1000";
for d in $DIRS
do
	cd $d 
	N=$d;
	rm "$N.csv";
	for f in MABCB7*.o*
	do
		P=$(grep "Nodes:" "$f" | awk '{print $2}');	
		WALL_TIME=$(grep "Wall" "$f" | awk '{print $6;}');
		V_TIME=$(grep "Serial" "$f" | awk '{print $6;}');
		RATIO=$(grep "Speedup" "$f" | awk '{print$3;}');
		echo "Nodes: $N Processes: $P Wall: $WALL_TIME V: $V_TIME R: $RATIO";
		echo "$P,$WALL_TIME,$V_TIME,$RATIO" >> "$N.csv";
	done
cd ../;
done
