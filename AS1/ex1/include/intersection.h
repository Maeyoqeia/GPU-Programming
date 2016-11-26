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
///  description: intersection with geometry
///
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _INTERSECTION_H_
#define _INTERSECTION_H_

// includes, project
#include "ray.h"
#include "intersection.h"
#include "point_light.h"
#include "material.h"

// includes, system
#include <memory>

class Intersection {

public:

  // constructor default
  Intersection();

  // destructor
  ~Intersection();

public:

  // compute color value
  float shade( const PointLight& light) const;

public:

  vec3 pos;
  vec3 w_out;
  vec3 n;

  std::shared_ptr<Material> mat;

private:

  // copy constructor
  Intersection( const Intersection&);

  // assignment operator
  Intersection& operator=( const Intersection&);

};

#endif // _INTERSECTION_H_
