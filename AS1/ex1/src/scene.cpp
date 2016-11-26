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
  for( int x = -10; x < 10; ++x) {
    for( int y = -10; y < 10; ++y) {
      for( int z = -10; z > -20; --z) {

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
Scene::traceRay( Ray& ray/*ray*/, Intersection& intersec/*intersec*/) const {
//just a single ray
    //multiple intersections
  // TODO: iterate over all objects and find closest intersection point
    bool flag = false;
    ray.t = std::numeric_limits<float>::max();
    float lambda = std::numeric_limits<float>::max();
    for( unsigned int i = 0; i < geometry.size(); ++i) {
        if(geometry[i]->intersect(ray, intersec))
            flag = true;                //if there is an intersection
    }
    return flag;
}
