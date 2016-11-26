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
///  description: base class for geometric primitives
///
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _GEOMETRY_H_
#define _GEOMETRY_H_

// includes, project
#include "ray.h"
#include "intersection.h"
#include "material.h"

// includes, system
#include <memory>


class Geometry {

public:

  // constructor default
  Geometry( std::shared_ptr<Material> m) :
    pos(),
    mat( m)
  { }

  // copy constructor
  Geometry( const Geometry& geo) :
    pos( geo.pos),
    mat( geo.mat)
  { }

  // destructor
  virtual ~Geometry() { };

public:

  //! Determine intersection point that is closer to ray.origin than current ray.t
  virtual bool intersect( Ray& ray, Intersection& intersec) const = 0;

private:

  // constructor, default
  Geometry();

  // assignment operator
  Geometry& operator=( const Geometry&);

public:

  // position of object
  vec3 pos;

  // material of object
  std::shared_ptr<Material> mat;

};

#endif // _GEOMETRY_H_
