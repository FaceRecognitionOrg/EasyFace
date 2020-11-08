#
# Author: zhongqianli <qianli_zh@qq.com>
# Date  : 2020-10-31
#

cmake_minimum_required(VERSION 2.8)

project(face_detector)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB_RECURSE ROOT_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

list(APPEND SRC_FILES ${ROOT_SRC_FILES})

# test
#add_executable(retinaface_test test/retinaface_test.cpp ${SRC_FILES})
#add_executable(face_detector_test example/face_detector_demo.cpp ${SRC_FILES})


# library
add_library(${PROJECT_NAME} STATIC ${SRC_FILES})
#add_library(${PROJECT_NAME} SHARED ${SRC_FILES})


set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# library test
add_executable(face_detector_demo example/face_detector_demo.cpp)
target_link_libraries(face_detector_demo PUBLIC ${PROJECT_NAME})

set_target_properties(face_detector_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)