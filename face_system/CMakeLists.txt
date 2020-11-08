#
# Author: zhongqianli <qianli_zh@qq.com>
# Date  : 2020-10-31
#

cmake_minimum_required(VERSION 2.8)

project(face_system)

include_directories(
    ${CMAKE_SOURCE_DIR}/face_detector/include
    ${CMAKE_SOURCE_DIR}/face_recognizer/include
)

add_executable(face_system_demo face_system_demo.cpp)
target_link_libraries(face_system_demo PUBLIC face_detector face_recognizer)

set_target_properties(face_system_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)