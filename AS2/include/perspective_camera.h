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
///  description: perspective camera
///
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _PERSPECTIVE_CAMERA_H_
#define _PERSPECTIVE_CAMERA_H_

// includes, project
#include "camera.h"

// includes, system
#include <limits>
#include <cmath>

// Simple scene description
class PerspectiveCamera : public Camera {

public:

  // constructor default
  PerspectiveCamera( int num_rows, int num_cols) :
    Camera( num_rows, num_cols)
  { }

  // destructor
  ~PerspectiveCamera() { };

public:

  //! generate ray for pixel
  void generateRay( int i_row, int i_col, Ray& ray);

};

#endif // _PERSPECTIVE_CAMERA_H_
