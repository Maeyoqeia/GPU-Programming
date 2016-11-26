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
///  description: ray
///
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _RAY_H_
#define _RAY_H_

// includes, system
#include <limits>

// includes, libraries
#include "glm/glm.hpp"
using namespace glm;


class Ray {

public:

  // constructor default
  Ray() :
    origin(),
    dir(),
    t( std::numeric_limits<float>::infinity())
  {}

  // destructor
  ~Ray() { };

public:

  vec3 operator()() {
    return origin + t * dir;
  }

  vec3 operator()( const float& tval) {
    return origin + tval * dir;
  }

public:

  // origin of ray
  vec3 origin;

  // direction of ray
  vec3 dir;

  // length along ray
  float t;

private:

  // copy constructor
  Ray( const Ray&);

  // assignment operator
  Ray& operator=( const Ray&);

};

#endif // _RAY_H_
