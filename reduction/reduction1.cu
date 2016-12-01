/// @file
////////////////////////////////////////////////////////////////////////////////////////////////////
///
/// Copyright (C) 2016/17      Christian Lessig, Otto-von-Guericke Universitaet Magdeburg
///
////////////////////////////////////////////////////////////////////////////////////////////////////
///
///  module     : lecture 6
///
///  author     : lessig@isg.cs.ovgu.de
///
///  project    : GPU Programming
///
///  description: reduction in Cuda
///
////////////////////////////////////////////////////////////////////////////////////////////////////

// includes, system
#include <iostream>
#include <vector>
#include <chrono>

typedef std::chrono::time_point<std::chrono::high_resolution_clock> tpoint;

// includes, project
#include "cuda_util.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
// initialize Cuda device
////////////////////////////////////////////////////////////////////////////////////////////////////
void
initDevice( int& device_handle, unsigned int& max_threads_per_block) {

  int deviceCount = 0;
  checkErrorsCuda( cudaGetDeviceCount(&deviceCount));

  if( 0 == deviceCount) {
    std::cerr << "initDevice() : No CUDA device found." << std::endl;
  }

  // one could implement more complex logic here to find the fastest device
  if( deviceCount > 1) {
    std::cerr << "initDevice() : Multiple CUDA devices found. Using first one." << std::endl;
  }

  // set the device
  checkErrorsCuda( cudaSetDevice( device_handle));

  cudaDeviceProp device_props;
  checkErrorsCuda( cudaGetDeviceProperties(&device_props, device_handle));
  max_threads_per_block = device_props.maxThreadsPerBlock;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// initialize device memory
////////////////////////////////////////////////////////////////////////////////////////////////////
void
initDeviceMemory( const std::vector<int>& data, int*& data_device, const unsigned int size) {

  // allocate device memory
  checkErrorsCuda( cudaMalloc((void **) &data_device, sizeof(int) * size));

  // copy device memory
  checkErrorsCuda( cudaMemcpy( data_device, &data[0], sizeof(int) * size, cudaMemcpyHostToDevice));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// initialize device memory
////////////////////////////////////////////////////////////////////////////////////////////////////
void
getResultDevice( const int* data_device, std::vector<int>& data, const unsigned int size) {

  checkErrorsCuda(cudaMemcpy( &data[0], data_device, sizeof(int) * size, cudaMemcpyDeviceToHost));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// free device memory
////////////////////////////////////////////////////////////////////////////////////////////////////
void
freeDeviceMemory( int*& data_device) {

  checkErrorsCuda( cudaFree( data_device));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// reduction
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void
reduction( int* data, unsigned int size) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = 1;

  while( stride < size) {
    if( 0 == (tid % (2*stride))) {
      data[tid] = data[tid] + data[tid+stride];
    }
    stride *= 2;
  }

}

////////////////////////////////////////////////////////////////////////////////////////////////////
// program entry point
////////////////////////////////////////////////////////////////////////////////////////////////////
int
main( int /*argc*/, char** /*argv*/ ) {

  // initialize device
  int device_handle = 0;
  unsigned int max_threads_per_block = 0;
  initDevice( device_handle, max_threads_per_block);

  // set up host memory
  // size is chosen so that two reduction steps would suffice
  const unsigned int size = 64 * max_threads_per_block * max_threads_per_block;
  std::vector<int> data( size);
  for( unsigned int i = 0; i < size; ++i) {
    data[i] = 1.0;
  }

  // initialize device memory
  int* data_device = nullptr;
  initDeviceMemory( data, data_device, size);

  // determine thread layout
  int num_threads_per_block = std::min( size, max_threads_per_block);
  int num_blocks = size / max_threads_per_block;
  if( 0 != size % max_threads_per_block) {
    num_blocks++;
  }

  reduction<<< num_blocks , num_threads_per_block >>>( data_device, size);
  checkLastCudaError( "Kernel launch failed.");

  getResultDevice( data_device, data, 1);
  int res = data[0];
  std::cerr << "Result = " << res << std::endl;

  // run again for timing
  cudaDeviceSynchronize();
  tpoint t_start = std::chrono::high_resolution_clock::now();
  for( unsigned int k = 0; k < 1024; ++k) {
    reduction<<< num_blocks , num_threads_per_block >>>( data_device, size);
  }
  cudaDeviceSynchronize();

  tpoint t_end = std::chrono::high_resolution_clock::now();
  double wall_clock = std::chrono::duration<double, std::milli>(t_end-t_start).count();
  std::cerr << "Execution time: " <<  wall_clock << " ms."<< std::endl;

  checkLastCudaError( "Kernel launch failed.");

  // clean up device memory
  freeDeviceMemory( data_device);

  return EXIT_SUCCESS;
}
