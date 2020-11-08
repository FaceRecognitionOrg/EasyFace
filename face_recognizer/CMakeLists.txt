#
# Author: zhongqianli <qianli_zh@qq.com>
# Date  : 2020-11-3
#

cmake_minimum_required(VERSION 2.8)

project(face_recognizer)

SET(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB_RECURSE ROOT_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

list(APPEND SRC_FILES ${ROOT_SRC_FILES})

# test
#add_executable(face_aligner_test test/face_aligner_test.cpp src/face_aligner.cpp)
#add_executable(mobilefacenet_test test/mobilefacenet_test.cpp src/mobilefacenet.cpp)
#add_executable(face_recognizer_impl_test test/face_recognizer_impl_test.cpp src/face_recognizer_impl.cpp src/face_aligner.cpp src/mobilefacenet.cpp)
#add_executable(face_recognizer_test example/face_recognizer_demo.cpp ${SRC_FILES})


# library
add_library(${PROJECT_NAME} STATIC ${SRC_FILES})
#add_library(${PROJECT_NAME} SHARED ${SRC_FILES})


set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# library test
add_executable(face_recognizer_demo example/face_recognizer_demo.cpp)
target_link_libraries(face_recognizer_demo PUBLIC ${PROJECT_NAME})

set_target_properties(face_recognizer_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)