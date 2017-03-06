#!/bin/bash

command="scuff-cas3D-wtrans.sh"

if [ $# -gt 0 ]
then
    L=$1
    scriptname="run-cas3D-L"$L$".script"
    echo "Submitting $command L="$L
    echo "$command "$L" "$gamma > $scriptname
    qsub -cwd $scriptname &
else
    echo "Enter Separation"
fi
