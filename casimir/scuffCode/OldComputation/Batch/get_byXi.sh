#!/bin/bash

dirs=$(ls | grep "Bead")
for dir in $dirs
do
    if [ -f $dir/*byXi ] 
    then
	cat $dir/*.byXi | grep -v "#"
    fi
done
