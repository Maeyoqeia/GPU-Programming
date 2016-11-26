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
///  description: Renderer
///
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _RENDERER_H_
#define _RENDERER_H_

// includes, project
#include "scene.h"
#include "camera.h"

// includes, system
#include <vector>


class Renderer {

public:

  // constructor default
  Renderer();

  // destructor
  virtual ~Renderer();

public:

  virtual void init( Camera* cam, Scene* scene);

  virtual void render();

  virtual float shade( const Intersection& intersec);

public:

  Camera*  cam;
  Scene*  scene;

};

#endif // _RENDERER_H_
