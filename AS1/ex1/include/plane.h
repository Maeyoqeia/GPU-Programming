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
///  description: plane
///
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _PLANE_H_
#define _PLANE_H_

// includes, project
#include "geometry.h"
#include "ray.h"
#include "intersection.h"


class Plane : public Geometry {

public:

  // constructor default
  Plane( std::shared_ptr<Material> m);

  // constructor, copy
  Plane( const Plane& plane);

  // destructor
  ~Plane();

public:

  void set( const vec3& normal, const vec3& tan1, const vec3& tan2, const vec2& tan_size);

  bool intersect( Ray& ray, Intersection& intersec) const;

  vec3 n;
  vec3 t1;
  vec3 t2;
  vec2 size;

private:


};

#endif // _PLANE_H_
