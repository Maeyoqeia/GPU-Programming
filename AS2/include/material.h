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
///  description: base class for materials
///
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _MATERIAL_H_
#define _MATERIAL_H_

// includes, project
#include "ray.h"

class Material {

public:

  // constructor default
  Material()  { }

  // destructor
  virtual ~Material() { };

public:

  //! Determine intersection point that is closer to ray.origin than current ray.t
  virtual float shade( const vec3& w_in, const vec3& w_out, const vec3& n) const = 0;

private:

  // copy constructor
  Material( const Material&);

  // assignment operator
  Material& operator=( const Material&);

};

#endif // _MATERIAL_H_
