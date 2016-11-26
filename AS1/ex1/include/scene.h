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
///  description: scene
///
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _SCENE_H_
#define _SCENE_H_

// includes, project
#include "ray.h"
#include "intersection.h"

// includes, system
#include <vector>

// declaration, forward
class Geometry;
class PointLight;


class Scene {

public:

  // constructor default
  Scene();

  // destructor
  ~Scene();

public:

  // initialize
  void init();

  // trace a ray through the scene and find the closest intersection point
  bool traceRay( Ray& ray, Intersection& intersec) const;

public:

  // scene geometry
  std::vector< Geometry* > geometry;

  // lights
  std::vector< PointLight* > lights;

private:

  // copy constructor
  Scene( const Scene&);

  // assignment operator
  Scene& operator=( const Scene&);

};

#endif // _GEOMETRY_H_
