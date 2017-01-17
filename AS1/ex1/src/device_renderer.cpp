/// @file
////////////////////////////////////////////////////////////////////////////////////////////////////
///
/// Copyright (C) 2016/17      Christian Lessig, Otto-von-Guericke Universitaet Magdeburg
///
////////////////////////////////////////////////////////////////////////////////////////////////////
///
///  module     : Exercise 1
///
///  author     : lessig@isg.cs.ovgu.de
///
///  project    : GPU Programming
///
///  description: device renderer
///
////////////////////////////////////////////////////////////////////////////////////////////////////

// includes, file
#include "device_renderer.h"
#include <chrono>
// includes, system
#include <iostream>

extern bool initDevice();
extern bool initDeviceMemory( const Scene& scene, const Image& image);
extern bool runDevice( const unsigned int n_rows, const unsigned int n_cols);
extern bool getImageDevice( Image& image);
extern void cleanupDevice();

////////////////////////////////////////////////////////////////////////////////////////////////////
// Constructor, default
////////////////////////////////////////////////////////////////////////////////////////////////////
DeviceRenderer::DeviceRenderer() :
  cam( nullptr),
  scene( nullptr)
{ }

////////////////////////////////////////////////////////////////////////////////////////////////////
// Destructor
////////////////////////////////////////////////////////////////////////////////////////////////////
DeviceRenderer::~DeviceRenderer() {
  delete cam;
  delete scene;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Initialize
////////////////////////////////////////////////////////////////////////////////////////////////////
void
DeviceRenderer::init( Camera* cam_cur, Scene* scene_cur) {

  cam = cam_cur;
  scene = scene_cur;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// add new object
////////////////////////////////////////////////////////////////////////////////////////////////////
/*virtual*/ void
DeviceRenderer::render() {

  auto t_start_all = std::chrono::high_resolution_clock::now();

  // initialize device
  initDevice();

  // set up data on the device
  initDeviceMemory( *scene, cam->image);

  // start timing
  auto t_start = std::chrono::high_resolution_clock::now();

  // launch kernels
  runDevice( cam->image.n_rows, cam->image.n_cols);

  // end timing
  auto t_end = std::chrono::high_resolution_clock::now();

  // copy result back to CPU
  getImageDevice( cam->image);

  // cleanup device memory
  cleanupDevice();


  // final time
  double wall_clock = std::chrono::duration<double, std::milli>(t_end-t_start).count();
  double wall_clock_all = std::chrono::duration<double, std::milli>(t_end-t_start_all).count();
  std::cerr << "Finished rendering: " <<  wall_clock << " ms."<< std::endl;
  std::cerr << "Total time including copying: " <<  wall_clock_all << " ms."<< std::endl;

}
