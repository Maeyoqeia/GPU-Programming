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
///  description: program main
///
////////////////////////////////////////////////////////////////////////////////////////////////////

// includes, system
#include <iostream>

// inclused, project
#include "renderer.h"
#include "parallel_renderer.h"
#include "scene.h"
#include "perspective_camera.h"
#include "device_renderer.h"

int
main( int /*argc*/, char** /**argv*/) {


  Camera* cam = new PerspectiveCamera( 512, 512);

  Scene* scene = new Scene();
  scene->init();

   // create renderer
 #ifdef DEVICE_COMPILE
   DeviceRenderer renderer;
   renderer.init( cam, scene);
 #else
   ParallelRenderer renderer;
   renderer.init( cam, scene);
 #endif

   // render
   renderer.render();

  // write output
  renderer.cam->image.write( "./images/im.pgm");

  return EXIT_SUCCESS;
}
