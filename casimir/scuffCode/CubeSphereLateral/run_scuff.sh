#!/bin/bash

filedir="/u/ki/kurinsky/GrattaGroup/opt_lev/casimir/scuffCode/"
tmpdir="/tmp/kurinsky/"
geofile="Bead.scuffgeo"
meshfiles="Sphere.geo Cube_Lateral.geo"
emeshfiles="Sphere.geo Cube_Edge.geo"
oldcubefile="Cube_Lateral.msh"
eoldcubefile="Cube_Edge.msh"
newcubefile="Cube.msh"

filestr="LateralBeadCube"

if [ $# -gt 4 ]
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
    W=$3
    gridding=$4
    T=$5

    if [ $W -ge 45 ]
    then
	echo "Using Edge Meshing"
	meshfiles=$emeshfiles
	oldcubefile=$eoldcubefile
    else
	echo "Using Lateral Meshing"
    fi

    #translation file name
    filebase=$filestr"_L-"$L"_W-"$W"_grid-"$gridding"_T-"$T

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

    echo "TRANS" $1 OBJECT Sphere DISP $W 0.0 $L >> $file
    cp $filedir/$geofile ./
    for mshfile in $meshfiles
    do
        cat $filedir/$mshfile | grep -v "grid =" | sed 's/grid/'$gridding'/' | grep -v "offset =" | sed 's/offset/'$W'/'
	cat $filedir/$mshfile | grep -v "grid =" | sed 's/grid/'$gridding'/' | grep -v "offset =" | sed 's/offset/'$W'/' > ./$mshfile
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
    echo "Calling Sequence: "$0" mode distance xwidth gridding temperature"
    exit 0
fi
