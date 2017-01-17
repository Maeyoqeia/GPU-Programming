cmake_minimum_required(VERSION 2.8)
project(AS3)

#set( CMAKE_CURRENT_BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR})
#set( ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
#set( SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(INCLUDE_DIRS
   ${CMAKE_CURRENT_SOURCE_DIR}/include
   /usr/local/include
)

IF (WIN32)
ELSE()
  set( CMAKE_CXX_FLAGS "-g -W -Wall -std=c++11 ")
ENDIF()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")

include_directories( ${INCLUDE_DIRS})

set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_executable( wavelet_transform "./src/main.cpp")

