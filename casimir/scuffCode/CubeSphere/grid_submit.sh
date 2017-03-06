#!/bin/bash

minlLog=1
maxlLog=1.5
dlLog=0.1

minG=0.3
maxG=0.4
dG=0.1

commands="./scuff-cas3d-PECtrans.sh ./scuff-cas3d-trans.sh"

for LLog in $(seq $minlLog $dlLog $maxlLog) 
do
    for G in $(seq $minG $dG $maxG)
    do
	L=$(awk 'BEGIN { print 10.0^'$LLog' }')
	for cmd in $commands
	do
	    echo "Submitting $cmd L="$L" G="$G
	    bsub -q xxl $cmd $L $G
	done
    done
done
