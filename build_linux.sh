#!/bin/bash
set -e

mkdir -p build-linux
#rm -rf build-linux/*
cd build-linux
cmake -DNCNN_VULKAN=OFF -DCMAKE_BUILD_TYPE=Release -DBUILD_SYSTEM=linux ..
make -j8
