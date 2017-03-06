#!/bin/bash

filedir="/u/ki/kurinsky/GrattaGroup/opt_lev/casimir/scuffCode/"
geofile="Bead.scuffgeo"
meshfiles="Sphere.geo Cube_Depth.geo"
oldcubefile="Cube_Depth.msh"
newcubefile="Cube.msh"

filestr="RatioBeadCube"

if [ $# -gt 4 ]
then
    L=$1
    gridding=$2
    ratio=$3
    depth=$4
    T=$5

    #translation file name
    filebase=$filestr"_L-"$L"_grid-"$gridding"_r-"$ratio"_d-"$depth"_T-"$T

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
        cat $filedir/$mshfile | grep -v "grid =" | sed 's/grid/'$gridding'/' | grep -v "gRatio =" | sed 's/gRatio/'$ratio'/' | grep -v "depth =" | sed 's/depth/'$depth'/'
	cat $filedir/$mshfile | grep -v "grid =" | sed 's/grid/'$gridding'/' | grep -v "gRatio =" | sed 's/gRatio/'$ratio'/' | grep -v "depth =" | sed 's/depth/'$depth'/' > ./$mshfile
        gmsh -2 $mshfile
    done

    mv $oldcubefile $newcubefile

    echo "scuff-cas3D --geometry $geofile --transfile $file --FileBase $filebase --Temperature $T --energy --zforce"
    scuff-analyze --geometry $geofile --transfile $file
    scuff-cas3D --geometry $geofile --transfile $file --FileBase $filebase --Temperature $T --energy --zforce

    rm -v *.msh
    rm -v *.geo

    tail -n 1 *.out >> ../results.txt

else
    echo "Not enough arguments"
    echo "Calling Sequence: "$0" distance gridding ratio depth temperature"
fi
