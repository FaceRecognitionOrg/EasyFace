#!/bin/bash
set -e

export PATH=$HOME/software/gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu/bin:$PATH

mkdir -p build-aarch64-linux-gnu
rm -rf build-aarch64-linux-gnu/*
cd build-aarch64-linux-gnu
cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/aarch64-linux-gnu.toolchain.cmake -DNCNN_VULKAN=OFF -DCMAKE_BUILD_TYPE=Release -DBUILD_SYSTEM=aarch64-linux-gnu ..
make -j4
