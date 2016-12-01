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
///  description: sphere
///
////////////////////////////////////////////////////////////////////////////////////////////////////

// includes, file
#include "perspective_camera.h"

// includes, project
#include "util.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
//! generate ray for pixel
////////////////////////////////////////////////////////////////////////////////////////////////////
void
PerspectiveCamera::generateRay( int i_row, int i_col, Ray& ray) {

  float pix_size_row = 1.0 / static_cast<float>(image.n_rows);
  float pix_size_col = 1.0 / static_cast<float>(image.n_cols);

  // assume [-0.5,0.5] x [-0.5,0.5] image plane with 1.0 focal length
  float y = static_cast<float>(i_row - image.n_rows/2) * pix_size_row + pix_size_row / 2.0;
  float x = static_cast<float>(i_col - image.n_cols/2) * pix_size_col + pix_size_col / 2.0;
  float z = -1.0;

  ray.origin = vec3( 0.0, 0.0, 0.0);
  ray.dir = vec3( x, y, z);
  ray.dir /= Util::norm( ray.dir);
  ray.t = std::numeric_limits<float>::infinity();

}
