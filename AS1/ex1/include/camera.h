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
///  description: base class for cameras
///
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _CAMERA_H_
#define _CAMERA_H_

// includes, project
#include "image.h"
#include "ray.h"

// Simple scene description
class Camera {

public:

  // constructor, user
  Camera( int num_rows, int num_cols) :
    image( num_rows, num_cols)
  { }

  // destructor
  virtual ~Camera() { };

public:

  //! generate ray for pixel
  virtual void generateRay( int i_row, int i_col, Ray& ray) = 0;

public:

  Image image;

private:

  // constructor, default
  Camera();

  // constructor, copy
  Camera( const Camera&);

  // assignment operator
  Camera& operator=( const Camera&);

};

#endif // _CAMERA_H_
