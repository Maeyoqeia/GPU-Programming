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
///  description: phong shading model
///
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _PHONG_H_
#define _PHONG_H_

// includes, file
#include "material.h"

class Phong : public Material {

public:

  // constructor default
  Phong();

  // destructor
  virtual ~Phong();

public:

  //! Determine intersection point that is closer to ray.origin than current ray.t
  float shade( const vec3& w_in, const vec3& w_out, const vec3& n) const;

public:

  float kd;

  float alpha;
  float ks;

private:

  // copy constructor
  Phong( const Phong&);

  // assignment operator
  Phong& operator=( const Phong&);

};

#endif // _PHONG_H_
