#!/bin/bash

filedir="/u/ki/kurinsky/GrattaGroup/opt_lev/casimir/scuffCode/"
geofile="Bead.scuffgeo"
meshfiles="Sphere.geo Cube_Aspect.geo"
oldcubefile="Cube_Aspect.msh"
newcubefile="Cube.msh"

filestr="AspectBeadCube"

if [ $# -gt 3 ]
then
    L=$1
    gridding=$2
    aspect=$3
    T=$4

    #translation file name
    filebase=$filestr"_L-"$L"_grid-"$gridding"_asp-"$aspect"_T-"$T

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
        cat $filedir/$mshfile | grep -v "grid =" | sed 's/grid/'$gridding'/' | grep -v "aspect =" | sed 's/aspect/'$aspect'/'
        cat $filedir/$mshfile | grep -v "grid =" | sed 's/grid/'$gridding'/' | grep -v "aspect =" | sed 's/aspect/'$aspect'/' > ./$mshfile
        gmsh -2 $mshfile
    done

    mv $oldcubefile $newcubefile

    echo "scuff-cas3D --geometry $geofile --transfile $file --FileBase $filebase --energy --zforce"
    scuff-analyze --geometry $geofile --transfile $file
    scuff-cas3D --geometry $geofile --transfile $file --FileBase $filebase --Temperature $T --energy --zforce

    rm -v *.msh
    rm -v *.geo

    tail -n 1 *.out >> ../results.txt

else
    echo "Not enough arguments"
    echo "Calling Sequence: "$0" distance gridding aspect temperature"
fi
