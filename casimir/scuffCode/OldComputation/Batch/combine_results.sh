#!/bin/bash

dirs=$(ls | grep "Bead")
for dir in $dirs
do
    if [ -f $dir/*out ] 
    then
	tail -n 1 $dir/*.out 
    fi
done
