#
# Author: zhongqianli <qianli_zh@qq.com>
# Date  : 2020-11-3
#

cmake_minimum_required(VERSION 2.8)

project(face_recognition)

# default: linux. others: aarch64-linux-gnu
option(BUILD_SYSTEM "build system")

if(${BUILD_SYSTEM} STREQUAL "aarch64-linux-gnu")
    set(LIBRARY_SUBDIR "aarch64-linux-lib")
else()
    set(LIBRARY_SUBDIR "linux-lib")
endif()

SET(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

option(NCNN_OPENMP "openmp support" ON)
option(NCNN_STDIO "load model from external file" ON)
option(NCNN_STRING "plain and verbose string" ON)
option(NCNN_INSTALL_SDK "Install ncnn library and headers" ON)
option(NCNN_OPENCV "minimal opencv structure emulation" OFF)
option(NCNN_BENCHMARK "print benchmark information for every layer" OFF)
option(NCNN_PIXEL "convert and resize from/to image pixel" ON)
option(NCNN_PIXEL_ROTATE "rotate image pixel orientation" OFF)
option(NCNN_CMAKE_VERBOSE "print verbose cmake messages" OFF)
option(NCNN_VULKAN "vulkan compute support" OFF)
option(NCNN_REQUANT "auto merge int8 quant and dequant" OFF)
option(NCNN_AVX2 "optimize x86 platform with avx2" OFF)

if(ANDROID OR IOS)
    option(NCNN_DISABLE_RTTI "disable rtti" ON)
    option(NCNN_BUILD_TOOLS "build tools" OFF)
else()
    option(NCNN_DISABLE_RTTI "disable rtti" OFF)
    option(NCNN_BUILD_TOOLS "build tools" ON)
endif()

if(ANDROID OR IOS OR LINUX)
    option(NCNN_DISABLE_EXCEPTION "disable exception" ON)
else()
    option(NCNN_DISABLE_EXCEPTION "disable exception" OFF)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ncnn/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/opencv/include)


link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ncnn/${LIBRARY_SUBDIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/opencv/${LIBRARY_SUBDIR}
)

link_libraries(
    ncnn
    opencv_core
    opencv_imgproc
    opencv_highgui
    opencv_imgcodecs
    opencv_videoio
)

add_subdirectory(face_detector)
add_subdirectory(face_recognizer)
add_subdirectory(face_system)