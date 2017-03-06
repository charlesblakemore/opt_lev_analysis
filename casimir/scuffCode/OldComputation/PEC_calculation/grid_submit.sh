#!/bin/bash

minlLog=-1
maxlLog=1
dlLog=0.1
command="./scuff-cas3D-wtrans.sh"

if [ $# -gt 0 ]
then
    if [ $1 -eq "1" ]
    then
	minlLog=-1
        maxlLog=-1
    fi
fi

for LLog in $(seq $minlLog $dlLog $maxlLog) 
do
    L=$(awk 'BEGIN { print 10.0^'$LLog' }')
    scriptname="run-cas3D-L"$L$".script"
    echo "Submitting $command L="$L
    bsub -q long $command $L
done
