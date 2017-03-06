#!/bin/bash

minl=5
maxl=30
dl=5

minw=50
maxw=50
dw=20

minG=0.4
maxG=0.5
dG=0.1

cmd="./run_scuff.sh"

echo "Testing Command"
$cmd > dump.txt
status=$?
rm dump.txt
if [ $status -ne 0 ]
then
    echo "Command Failed"
    exit 1
fi
echo "Command Successful"

for mode in 0 1
do
    for temp in 300 0
    do
	for L in 1 $(seq $minl $dl $maxl) 
	do
	    for W in $(seq $minw $dw $maxw)
	    do
		for G in $(seq $minG $dG $maxG)
		do
		    echo "Submitting $cmd mode="$mode" L="$L" W="$W" G="$G" T="$temp
		    bsub -q xxl $cmd $mode $L $W $G $temp
		done
	    done
	done
    done
done
