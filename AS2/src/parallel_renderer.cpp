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
///  description: parallel renderer
///
////////////////////////////////////////////////////////////////////////////////////////////////////

// includes, file
#include "parallel_renderer.h"

// includes, system
#include <iostream>
#include <thread>
#include <chrono>


////////////////////////////////////////////////////////////////////////////////////////////////////
// Constructor, default
////////////////////////////////////////////////////////////////////////////////////////////////////
ParallelRenderer::ParallelRenderer() :
  Renderer()
{ }

////////////////////////////////////////////////////////////////////////////////////////////////////
// Destructor
////////////////////////////////////////////////////////////////////////////////////////////////////
ParallelRenderer::~ParallelRenderer() { }

////////////////////////////////////////////////////////////////////////////////////////////////////
// Initialize
////////////////////////////////////////////////////////////////////////////////////////////////////
void
ParallelRenderer::init( Camera* cam_cur, Scene* scene_cur) {

  // TODO: determine available hardware capabilities
    unsigned num_threads = std::thread::hardware_concurrency();

  std::cout << "ParallelRenderer::init() : Using " << num_threads << " threads." << std::endl;

  Renderer::init( cam_cur, scene_cur);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Initialize
////////////////////////////////////////////////////////////////////////////////////////////////////
void
ParallelRenderer::init( Camera* cam_cur, Scene* scene_cur, const unsigned int nt) {

  num_threads = nt;
  Renderer::init( cam_cur, scene_cur);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// render a single tile
////////////////////////////////////////////////////////////////////////////////////////////////////
void
ParallelRenderer::renderTile( const ivec2 tid/*tid*/, const ivec2& num_threads/*num_threads*/) {

  // TODO: render tile
    unsigned int deltax = this->cam->image.n_rows / num_threads[0];
   unsigned int deltay = this->cam->image.n_rows / num_threads[1];

      unsigned int ix1 = tid[0] * deltax;
      unsigned int ix2 = (tid[0]+1) * deltax;
      unsigned int iy1 = tid[1] * deltay;
      unsigned int iy2 = (tid[1]+1) * deltay;
 Ray* ray = new Ray();
 Intersection* intersec = new Intersection();
      for (unsigned int i = ix1; i < ix2; ++i) {
          for (unsigned int j = iy1; j < iy2; ++j) {

              this->cam->generateRay(i,j,*ray);

              if(this->scene->traceRay(*ray, *intersec))
              {
                  this->cam->image(i,j) =  shade(*intersec);
              }

          }

      }
      delete(ray);
      delete(intersec);

}

////////////////////////////////////////////////////////////////////////////////////////////////////
// render
////////////////////////////////////////////////////////////////////////////////////////////////////
void
ParallelRenderer::render() {

  // TODO: determine tile layout

    int lg = std::log2(num_threads);
    int x_pow = lg/2;
    int y_pow = lg-x_pow;
    num_tiles[0] = pow(2,x_pow);
    num_tiles[1] = pow(2,y_pow);

  std::cerr << "ParallelRenderer::render() : running with "
            << num_tiles[0] << " x " << num_tiles[1] << "." << std::endl;

  // start timing
  auto t_start = std::chrono::high_resolution_clock::now();

  // TODO: spawn threads
  const ivec2 numthreads(num_tiles[0],num_tiles[1]);
  std::vector< std::thread > threads;
      for( unsigned int i = 0; i < num_tiles[0]; ++i) {
          for (int j = 0; j < num_tiles[1]; ++j) {
              const ivec2 tid(i,j);

              threads.push_back( std::thread( &ParallelRenderer::renderTile, this, tid, numthreads));
          }

      }

      for( unsigned int i = 0; i < num_tiles[0]*num_tiles[1]; ++i) {
        threads[i].join();
      }


  // end timing
  auto t_end = std::chrono::high_resolution_clock::now();
  // final time
  double wall_clock = std::chrono::duration<double, std::milli>(t_end-t_start).count();
  std::cerr << "Finished rendering: " <<  wall_clock << " ms."<< std::endl;
}
