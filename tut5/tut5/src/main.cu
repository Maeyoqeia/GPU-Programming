/// @file
////////////////////////////////////////////////////////////////////////////////////////////////////
///
/// Copyright (C) 2016/17      Christian Lessig, Otto-von-Guericke Universitaet Magdeburg
///
////////////////////////////////////////////////////////////////////////////////////////////////////
///
///  module     : tutorial 5
///
///  author     : lessig@isg.cs.ovgu.de
///
///  project    : GPU Programming
///
///  description: CUDA convolution
///
////////////////////////////////////////////////////////////////////////////////////////////////////

// includes, system
#include <iostream>
#include <algorithm>
#include <chrono>

typedef std::chrono::time_point<std::chrono::high_resolution_clock> tpoint;

// includes, project
#include "cuda_util.h"
#include "kernel_separable.h"
#include "image.h"

// host implementation
extern void
convSeparableHost( float* kdata, const int& kernel_supp, const Image& image, Image& image_conv);

////////////////////////////////////////////////////////////////////////////////////////////////////
// convolution
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__
void
convSeparable1( float* kernel, const int kernel_supp_half,
                float* image, float* image_conv, const unsigned int image_size) {

    //gridDim - wie viele blöcke gibt es
    int pix_r = thread.Idx.x+blockIdx.x*blockDim.x; //id innerhalb des thread blocks, zwischen 0 und 32=sqrt(2^10)
    int pix_c = threadIdx.y + blockIdx.y * blockDim.y;

    float weight_row = 0.0;
        float weight = 0.0;

        int ik = 0;
        int jk = 0;

        image_conv[pix_r * image_size + col] = 0.0;

        for( int i = pix_r - SuppHalf; i <= pix_r + SuppHalf; ++i, ++ik) {
          weight_row = kernel[ik];
          jk = 0;
          for( int j = pix_c - SuppHalf; j <= pix_c + SuppHalf; ++j, ++jk) {

            if( ( i < 0 || j < 0) || (i >= image_size) || (j >= image_size)) {
              continue;
            }

            weight = weight_row * kernel[jk];
            image_conv[pix_r * image_size + pix_c] += weight * image[i * image_size + j];
          }
        }
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

  const int kernel_supp = 5;
  const int kernel_supp_half = kernel_supp / 2;
  float kdata[] = {0.0103339f, 0.207561f, 0.56421f, 0.207561f, 0.0103339f};

  Image image( "../images/im.pgm");

  Image image_conv( image.n_rows, image.n_cols);
  convSeparableHost( kdata, kernel_supp_half, image, image_conv);
  image_conv.write( "../images/im_conv_host.pgm");


  // check execution environment
  int device_handle = 0;
  int max_threads_per_block = 0;
  if( ! initDevice( device_handle, max_threads_per_block)) {
    return EXIT_FAILURE;
  }

  // initialize memory
  float* kernel_device = nullptr;
  float* image_device = nullptr;
  float* image_conv_device = nullptr;

  // allocate device memory
  checkErrorsCuda( cudaMalloc((void **) &kernel_device, sizeof(float) * kernel_supp));
  checkErrorsCuda( cudaMalloc((void **) &image_device, sizeof(float) * image.n_cols * image.n_rows));
  checkErrorsCuda( cudaMalloc((void **) &image_conv_device, sizeof(float) * image.n_cols * image.n_rows));

  // copy device memory
  checkErrorsCuda( cudaMemcpy( (void*) kernel_device, kdata,
                                sizeof(float) * kernel_supp,
                                cudaMemcpyHostToDevice ));
  checkErrorsCuda( cudaMemcpy( (void*) image_device, &(image.data[0]),
                                sizeof(float) * image.n_cols * image.n_rows,
                                cudaMemcpyHostToDevice ));

  // determine thread layout
  unsigned int sqrt_max_threads = (unsigned int) std::sqrt(max_threads_per_block);
  dim3 num_threads_per_block(sqrt_max_threads,sqrt_max_threads) = num_blocks;
  dim3 num_blocks;
  std::cout << "num_blocks = " << num_blocks.x << " / " << num_blocks.y << std::endl;
  std::cout << "num_threads_per_block = " << num_threads_per_block.x << " / "
                                          << num_threads_per_block.y << std::endl;

  // run kernel
  assert( image.n_rows == image.n_cols);
  tpoint t_start = std::chrono::high_resolution_clock::now();
  convSeparable1<<< num_blocks , num_threads_per_block >>>( kernel_device, kernel_supp_half, image_device,
                                                           image_conv_device, image.n_rows);

  tpoint t_end = std::chrono::high_resolution_clock::now();
  double wall_clock = std::chrono::duration<double, std::milli>(t_end-t_start).count();
  std::cerr << "Execution time: " <<  wall_clock << " ms."<< std::endl;

  checkLastCudaError("Kernel execution failed");
  cudaDeviceSynchronize();

  // copy result back to host
  checkErrorsCuda( cudaMemcpy( &image_conv.data[0], image_conv_device,
                               sizeof(float) * image.n_cols * image.n_rows,
                               cudaMemcpyDeviceToHost ));
  // write result
  image_conv.write( "../images/im_conv_device.pgm");

  // clean up device memory
  checkErrorsCuda( cudaFree( kernel_device));
  checkErrorsCuda( cudaFree( image_device));
  checkErrorsCuda( cudaFree( image_conv_device));

  return EXIT_SUCCESS;
}
