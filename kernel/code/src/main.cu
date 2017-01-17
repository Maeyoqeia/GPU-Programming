/// @file
////////////////////////////////////////////////////////////////////////////////////////////////////
///
/// Copyright (C) 2016/17      Christian Lessig, Otto-von-Guericke Universitaet Magdeburg
///
////////////////////////////////////////////////////////////////////////////////////////////////////
///
///  module     : tutorial 6
///
///  author     : lessig@isg.cs.ovgu.de
///
///  project    : GPU Programming
///
///  description: CUDA matrix transpose
///
////////////////////////////////////////////////////////////////////////////////////////////////////

// includes, system
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cassert>

typedef std::chrono::time_point<std::chrono::high_resolution_clock> tpoint;

// includes, project
#include "cuda_util.h"

const unsigned int Tile_Size = 32;

////////////////////////////////////////////////////////////////////////////////////////////////////
//! Matrix transpose (no bank conflicts)
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__
void
transposeMatrix3( float* data_in, float* data_out, unsigned int mat_size) {

  int tid_col = blockIdx.x * blockDim.x + threadIdx.x;
  int tid_row = blockIdx.y * blockDim.y + threadIdx.y;

  __shared__ float sdata[Tile_Size][Tile_Size+1];
  sdata[threadIdx.y][threadIdx.x] = data_in[tid_row * mat_size + tid_col];
  __syncthreads();

  tid_col = blockIdx.y * blockDim.x + threadIdx.x;
  tid_row = blockIdx.x * blockDim.y + threadIdx.y;

  data_out[tid_row * mat_size + tid_col] = sdata[threadIdx.x][threadIdx.y];
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//! Matrix transpose (shared memory to ensure coalesced reads and writes)
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__
void
transposeMatrix2( float* data_in, float* data_out, unsigned int mat_size) {

  int tid_col = blockIdx.x * blockDim.x + threadIdx.x;
  int tid_row = blockIdx.y * blockDim.y + threadIdx.y;

  __shared__ float sdata[Tile_Size][Tile_Size];
  sdata[threadIdx.x][threadIdx.y] = data_in[tid_row * mat_size + tid_col];
  __syncthreads();

  tid_col = blockIdx.y * blockDim.y + threadIdx.x;
  tid_row = blockIdx.x * blockDim.y + threadIdx.y;

  data_out[tid_row * mat_size + tid_col] = sdata[threadIdx.y][threadIdx.x];
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//! Matrix transpose (naive implementation)
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__
void
transposeMatrix1( float* data_in, float* data_out, unsigned int mat_size) {

  int tid_col = blockIdx.x * blockDim.x + threadIdx.x;
  int tid_row = blockIdx.y * blockDim.y + threadIdx.y;

  data_out[tid_col * mat_size + tid_row] = data_in[tid_row * mat_size + tid_col];
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//! Matrix copy (as reference for maximal attainable performance)
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__
void
copyMatrix( float* data_in, float* data_out, unsigned int mat_size) {

  int tid_col = blockIdx.x * blockDim.x + threadIdx.x;
  int tid_row = blockIdx.y * blockDim.y + threadIdx.y;

  data_out[tid_row * mat_size + tid_col] = data_in[tid_row * mat_size + tid_col];
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// initialize Cuda device
////////////////////////////////////////////////////////////////////////////////////////////////////
bool
initDevice( int& device_handle, int& max_threads_per_block) {

  int deviceCount = 0;
  checkErrorsCuda( cudaGetDeviceCount(&deviceCount));

  if( 0 == deviceCount) {
    std::cerr << "initDevice() : No CUDA device found." << std::endl;
    return false;
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

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// program entry point
////////////////////////////////////////////////////////////////////////////////////////////////////
int
main( int /*argc*/, char** /*argv*/ ) {

  // check execution environment
  int device_handle = 0;
  int max_threads_per_block = 0;
  if( ! initDevice( device_handle, max_threads_per_block)) {
    return EXIT_FAILURE;
  }

  const int mat_size = 8192;
  // input matrix
  std::vector<float> mat_in( mat_size * mat_size);
  std::generate( mat_in.begin(), mat_in.end(), std::rand);

  // initialize memory
  float* mat_in_device = nullptr;
  float* mat_out_device = nullptr;

  // allocate device memory
  checkErrorsCuda( cudaMalloc((void **) &mat_in_device, sizeof(float) * mat_size * mat_size));
  checkErrorsCuda( cudaMalloc((void **) &mat_out_device, sizeof(float) * mat_size * mat_size));

  // copy device memory
  checkErrorsCuda( cudaMemcpy( (void*) mat_in_device, mat_in.data(),
                                sizeof(float) * mat_size * mat_size,
                                cudaMemcpyHostToDevice ));

  // determine thread layout
  int max_threads_per_block_sqrt = std::sqrt( max_threads_per_block);
  assert( max_threads_per_block_sqrt * max_threads_per_block_sqrt == max_threads_per_block);
  assert( max_threads_per_block_sqrt == Tile_Size);
  dim3 num_threads( std::min( mat_size, max_threads_per_block_sqrt),
                    std::min( mat_size, max_threads_per_block_sqrt) );
  dim3 num_blocks( mat_size / num_threads.x, mat_size / num_threads.y);
  num_blocks.x += ( 0 == num_blocks.x) ? 1 : 0;
  num_blocks.y += ( 0 == num_blocks.y) ? 1 : 0;

  std::cout << "num_blocks = " << num_blocks.x << " / " << num_blocks.y << std::endl;
  std::cout << "num_threads_per_block = " << num_threads.x << " / "
                                          << num_threads.y << std::endl;

  // run kernel
  cudaDeviceSynchronize();
  tpoint t_start = std::chrono::high_resolution_clock::now();

#if 0
  transposeMatrix1<<<num_blocks, num_threads>>>( mat_in_device, mat_out_device, mat_size);
#endif
#if 0
  transposeMatrix2<<<num_blocks, num_threads>>>( mat_in_device, mat_out_device, mat_size);
#endif
#if 1
  transposeMatrix3<<<num_blocks, num_threads>>>( mat_in_device, mat_out_device, mat_size);
#endif

  cudaDeviceSynchronize();
  tpoint t_end = std::chrono::high_resolution_clock::now();
  double wall_clock = std::chrono::duration<double, std::milli>(t_end-t_start).count();
  std::cerr << "Execution time: " <<  wall_clock << " ms."<< std::endl;

  checkLastCudaError("Kernel execution failed");

  // copy result back to host
  std::vector<float> mat_out( mat_size * mat_size);
  checkErrorsCuda( cudaMemcpy( mat_out.data(), mat_out_device,
                               sizeof(float) * mat_size * mat_size,
                               cudaMemcpyDeviceToHost ));

#if 1
  // check result
  for( unsigned int row = 0; row < mat_size; ++row) {
    for( unsigned int col = 0; col < mat_size; ++col) {
      if( mat_out[col * mat_size + row] != mat_in[row * mat_size + col]) {
        std::cerr << "Transpose error at (" << row << "," << col << ")" << std::endl;
      }
    }
  }
#endif

  // clean up device memory
  checkErrorsCuda( cudaFree( mat_in_device));
  checkErrorsCuda( cudaFree( mat_out_device));

  return EXIT_SUCCESS;
}
