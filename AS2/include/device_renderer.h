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

#ifndef _DEVICE_RENDERER_H_
#define _DEVICE_RENDERER_H_

// includes, file
#include "renderer.h"
#include <chrono>
// includes, project
#include "scene.h"
#include "camera.h"

// Simple DeviceRenderer description
class DeviceRenderer : public Renderer {

public:

  // constructor default
  DeviceRenderer();

  // destructor
  ~DeviceRenderer();

public:

  void init( Camera* cam, Scene* scene);

  void render();

public:

  Camera*  cam;
  Scene*  scene;

};

#endif // _DEVICE_RENDERER_H_
