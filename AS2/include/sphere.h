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

#ifndef _SPHERE_H_
#define _SPHERE_H_

// includes, project
#include "geometry.h"
#include "ray.h"
#include "intersection.h"


class Sphere : public Geometry {

public:

  // constructor default
  Sphere( std::shared_ptr<Material> m);

  // constructor, copy
  Sphere( const Sphere& sphere);

  // destructor
  ~Sphere();

public:

  bool intersect( Ray& ray, Intersection& intersec) const;

public:

  float r;

};

#endif // _SPHERE_H_
