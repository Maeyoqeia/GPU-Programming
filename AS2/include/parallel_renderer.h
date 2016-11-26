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

#ifndef _PARALLEL_RENDERER_H_
#define _PARALLEL_RENDERER_H_

// includes, project
#include "scene.h"
#include "camera.h"
#include "renderer.h"

// includes, system
#include <mutex>

class ParallelRenderer : public Renderer {

public:

  // constructor default
  ParallelRenderer();

  // destructor
  ~ParallelRenderer();

public:

  // initialize; dynamically determine the number of threads that is appropriate for the
  // hardware at hand
  void init( Camera* cam, Scene* scene);

  // initialize
  void init( Camera* cam, Scene* scene, unsigned int num_threads);

  void render();

private:

  void renderTile( const ivec2 tid, const ivec2& num_threads);

public:

  // number of threads to be used
  int num_threads;

  // number of tiles for parallelization
  ivec2 num_tiles;

  // mutex to control access to the image
  std::mutex mutex_tiles;
};

#endif // _PARALLEL_RENDERER_H_
