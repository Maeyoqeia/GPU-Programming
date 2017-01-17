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

// includes, file
#include "scene.h"

// includes, project
#include "geometry.h"
#include "sphere.h"
#include "plane.h"
#include "point_light.h"
#include "phong.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
// Constructor, default
////////////////////////////////////////////////////////////////////////////////////////////////////
Scene::Scene() :
  geometry()
{ }

////////////////////////////////////////////////////////////////////////////////////////////////////
// Destructor
////////////////////////////////////////////////////////////////////////////////////////////////////
Scene::~Scene() {
  // clean up
  for( unsigned int i = 0; i < geometry.size(); ++i) {
    delete geometry[i];
  }
  for (unsigned int i = 0; i < lights.size(); ++i) {
    delete lights[i];
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Create scene
////////////////////////////////////////////////////////////////////////////////////////////////////
void
Scene::init() {

  // set up phong material
  Phong* phong = new Phong();
  phong->kd = 1.0;
  phong->ks = 0.5;
  phong->alpha = 32.0;
  std::shared_ptr<Material> mat( phong);

  // grid of spheres
  for( int x = -8; x < 8; ++x) {
    for( int y = -8; y < 8; ++y) {
      for( int z = -10; z > -16; --z) {

        Sphere* sphere = new Sphere( mat);
        sphere->pos = vec3( static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
        sphere->r = 0.25;
        // add new object
        geometry.push_back( sphere);
      }
    }
  }

  // base plane
  Plane* plane = new Plane( mat);
  plane->pos = vec3( 0.0, -2.0, 0.0);
  plane->set( vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), vec3(0.0, 0.0, -1.0), vec2(100.0, 100.0));
  geometry.push_back( plane);

  // add lights

  PointLight* light1 = new PointLight();
  light1->ell = 255.0;
  light1->pos = vec3( 3.0, 3.0, -5.0);
  lights.push_back( light1);

  PointLight* light2 = new PointLight();
  light2->ell = 64.0;
  light2->pos = vec3( 0.0, 10.0, -15.0);
  lights.push_back( light2);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// trace ray through scene
////////////////////////////////////////////////////////////////////////////////////////////////////
bool
Scene::traceRay( Ray& ray, Intersection& intersec) const {

  bool has_hit = false;
  for( auto object : geometry) {
    has_hit |= object->intersect( ray, intersec);
  }

  return has_hit;
}
