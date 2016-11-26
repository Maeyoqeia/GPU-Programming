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
///  description: point light
///
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _LIGHT_H_
#define _LIGHT_H_

// includes, libraries
#include "glm/glm.hpp"
using namespace glm;

// Simple scene description
class PointLight {

public:

  // constructor, user
  PointLight() :
    ell( 0.0),
    pos()
   { }

  // destructor
  ~PointLight() { };

public:

  // intensity
  float ell;

  // position
  vec3 pos;

};

#endif // _LIGHT_H_
