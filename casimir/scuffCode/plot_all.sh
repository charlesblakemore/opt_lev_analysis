#!/bin/bash

pushd CubeSphere
python compare.py
popd

pushd CubeSphereAspect
python width.py
python width_finite.py
popd

pushd CubeSphereBest
python compare.py
python prediction.py
popd

pushd CubeSphereDepth
python width.py
python width_finite.py
popd

pushd CubeSphereLateral
python trend.py
python trend_finite.py
popd
