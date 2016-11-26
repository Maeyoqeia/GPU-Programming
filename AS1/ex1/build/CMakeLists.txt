# CMakeLists.txt
cmake_minimum_required(VERSION 2.8)
project( ex1)

set( CMAKE_CURRENT_BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set( ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set( SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src)

set(SOURCE
   ${SRC_DIR}/main.cpp
   ${SRC_DIR}/scene.cpp
   ${SRC_DIR}/sphere.cpp
   ${SRC_DIR}/plane.cpp
   ${SRC_DIR}/renderer.cpp
   ${SRC_DIR}/parallel_renderer.cpp
   ${SRC_DIR}/intersection.cpp
   ${SRC_DIR}/image.cpp
   ${SRC_DIR}/perspective_camera.cpp
   ${SRC_DIR}/phong.cpp
)

set(CUDA_SOURCE
   ${SOURCE}
   ${SRC_DIR}/device_renderer.cpp
   ${SRC_DIR}/device_renderer.cu
)

set(INCLUDE_DIRS
   ${CMAKE_CURRENT_SOURCE_DIR}/../include
   ${CMAKE_CURRENT_SOURCE_DIR}/../inl
   /usr/local/include
)

IF (WIN32)
ELSE()
  set( CMAKE_CXX_FLAGS "-g -W -Wall -std=c++11 -pthread")
ENDIF()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")

include_directories( ${INCLUDE_DIRS})

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_executable( raycaster ${SOURCE})

# cuda_add_executable( raycaster_device ${CUDA_SOURCE})
