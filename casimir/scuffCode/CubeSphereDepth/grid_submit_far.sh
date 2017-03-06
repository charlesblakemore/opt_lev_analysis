#!/bin/bash

minl=20
maxl=20
dl=2.5

minG=0.4
maxG=0.4
dG=0.1

temp="0 300"
ratio="5.0 10.0 20.0"
depth="1 2 3 5"

commands="./trans_width_temp.sh ./trans_width_temp_PEC.sh"

for L in $(seq $minl $dl $maxl) 
do
    for G in $(seq $minG $dG $maxG)
    do
	for r in $ratio
	do
	    for d in $depth
	    do
		for T in $temp
		do
		    for cmd in $commands
		    do
			echo "Submitting $cmd L="$L" G="$G" R="$r" D="$d" T="$T
			bsub -q xxl $cmd $L $G $r $d $T
		    done
		done
	    done
	done
    done
done
