#!/bin/bash

filedir="/u/ki/kurinsky/GrattaGroup/opt_lev/casimir/scuffCode/"
tmpdir="/tmp/kurinsky/"
geofile="Bead.scuffgeo"
meshfiles="Sphere.geo Cube_Best.geo"
oldcubefile="Cube_Best.msh"
newcubefile="Cube.msh"

filestr="RatioBeadCube"

if [ $# -gt 3 ]
then
    if [ $1 -eq 0 ]
    then
	echo "Simulating Infinite Conductivity"
	geofile="Bead_PEC.scuffgeo"
	filestr=$filestr"PEC"
    elif [ $1 -eq 1 ]
    then
	echo "Simulating Finite Conductivity"
	geofile="Bead.scuffgeo"
    else
	echo "Invalid first argument"
	exit
    fi

    L=$2
    gridding=$3
    T=$4

    #translation file name
    filebase=$filestr"_L-"$L"_grid-"$gridding"_T-"$T

    mkdir -p $tmpdir
    pushd $tmpdir

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

    echo "TRANS" $1 OBJECT Sphere DISP 0.0 0.0 $L >> $file
    cp $filedir/$geofile ./
    for mshfile in $meshfiles
    do
        cat $filedir/$mshfile | grep -v "grid =" | sed 's/grid/'$gridding'/' 
	cat $filedir/$mshfile | grep -v "grid =" | sed 's/grid/'$gridding'/' > ./$mshfile
        gmsh -2 $mshfile
    done

    mv $oldcubefile $newcubefile

    echo "scuff-cas3D --geometry $geofile --transfile $file --FileBase $filebase --Temperature $T --energy --zforce"
    scuff-analyze --geometry $geofile --transfile $file
    rm -v *.pp
    scuff-cas3D --geometry $geofile --transfile $file --FileBase $filebase --Temperature $T --energy --zforce
    if [ $? -eq 0 ]
    then
	echo "Successful!"
    fi

    rm -v *.msh
    rm -v *.geo

    popd
    mv $tmpdir/$filebase ./$filebase

    exit 0
else
    echo "Not enough arguments"
    echo "Calling Sequence: "$0" mode distance gridding temperature"
    exit 0
fi
