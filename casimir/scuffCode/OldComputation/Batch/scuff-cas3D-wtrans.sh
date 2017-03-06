#!/bin/bash

filedir="../../"
geofile="Bead.scuffgeo"
meshfiles="Sphere.msh Surface.msh"

filestr="Bead_L-"

if [ $# -gt 0 ]
then
    #translation file name
    filebase=$filestr$1
    
    #translation range/step in microns
    minimum=$1
    if [ $# -gt 1 ]
    then
	maximum=$2
	filebase=$filestr$1"-"$2
    else
	maximum=$1
    fi
    
    if [ $# -gt 2 ]
    then
	step=$3
	filebase=$filestr$1"-"$2"-d"$3
    else
	step=1.0
    fi    

    if [ -d $filebase ]
    then
	rm -rvf $filebase
    fi
    mkdir $filebase
    cd $filebase
    
    file=$filebase".trans"
    if [ -f $file ]
    then
        printf "Removing Old Trans File: "
        rm -rv $file
    fi

    for i in $(seq $minimum $step $maximum) 
    do
	echo "TRANS" $i OBJECT Sphere DISP 0.0 0.0 $i >> $file
    done
    cp $filedir/$geofile ./
    for mshfile in $meshfiles
    do
	cp $filedir/$mshfile ./
    done
    
    scuff-cas3D --geometry $geofile --transfile $file --FileBase $filebase --energy --zforce

else
    echo "Not enough arguments"
    echo "Calling Sequence: "$0" mimumum [maximum] [step]"
fi
