#!/bin/bash

minl=10
maxl=30
dl=2.5

minG=0.3
maxG=0.4
dG=0.1

temp="0 300"
aspect="15.0 20.0"

commands="./trans_width_temp.sh ./trans_width_temp_PEC.sh"

for L in $(seq $minl $dl $maxl) 
do
    for G in $(seq $minG $dG $maxG)
    do
	for asp in $aspect
	do
	    for T in $temp
	    do
		for cmd in $commands
		do
		    echo "Submitting $cmd L="$L" G="$G" A="$asp" T="$T
	            bsub -q xxl $cmd $L $G $asp $T
		done
	    done
	done
    done
done
