
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
///  description: Cuda implementation
///
////////////////////////////////////////////////////////////////////////////////////////////////////

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>

// include, project
#include "cuda_util.h"
#include "scene.h"
#include "image.h"
#include "sphere.h"
#include "plane.h"
#include "phong.h"
#include "util.h"
#include "device_renderer.cuh"

// includes, system
#include <iostream>
#include <vector>
#include <map>
#include <chrono>
#include <algorithm>


namespace {
int max_threads_per_block;
struct DevicePLight{
    float ill; //intensity illumination
    vec3 pos;
};


struct DeviceImage{
    float* img;
    int n_cols;
    int n_rows;
};
struct DeviceRay{
    vec3 dir;
    float t;
};

struct DeviceSphere
{
vec3 center;
float r;
};

struct DevicePlane
{
vec3 pos;
vec3 normal;
vec3 t1;
vec3 t2;
vec2 size;
};

struct DeviceIntersection
{
    vec3 pos;
    vec3 w_out;
    vec3 normal;
};
struct PhongMaterial{
    float ks;
    float kd;
    float alpha;
};

struct DeviceScene{
    DevicePLight* lights;
    DevicePlane* planes;
    DeviceSphere* spheres;
    unsigned int numLights;
    unsigned int numPlanes;
    unsigned int numSpheres;
    PhongMaterial material;
};
struct DeviceData{
    DeviceScene scene;
    DeviceImage img;
}data;
}

__device__
float
norm2( const vec3& vec) {
  return vec.x*vec.x + vec.y*vec.y + vec.z*vec.z;
}

__device__
float
norm( const vec3& vec) {
  return sqrt( norm2(vec));
}

__device__
float
dot( const vec3& vec1, const vec3& vec2) {
  return vec1.x*vec2.x + vec1.y*vec2.y + vec1.z*vec2.z;
}


__device__
bool
intersect_sphere( DeviceRay& ray , DeviceIntersection& intersec, DeviceSphere& sphere){

  // TODO: compute intersection point
    vec3 origin(0,0,0);
    vec3 dir = ray.dir;
    //test whether discriminant is less than zero

    vec3 a = origin-sphere.center;
    float b = (dot(dir, a));

    float dis = b*b - (norm(a)*norm(a)) + sphere.r*sphere.r;

        if(dis >= 0){
            float t01 = -b + sqrt(dis);
            float t02 = -b - sqrt(dis);
            float t = min(t01,t02);
            if(t < ray.t && t > 0){
                ray.t =t;
                intersec.pos = dir*t; //ray.origin = 000
                intersec.normal = intersec.pos - sphere.center;
                intersec.normal /= norm(intersec.normal);
                intersec.w_out = origin - intersec.pos;
                intersec.w_out /= norm(intersec.w_out);
                return true;
            }
         }

        return false;


}
__device__
bool
intersect_plane( DeviceRay& ray, DeviceIntersection& intersec,DevicePlane plane) {

   float dotProd=dot(ray.dir,plane.normal);
    vec3 origin(0,0,0);
   //return if ray is nearly parallel to plane
   if(abs(dotProd) < std::numeric_limits<float>::epsilon())
       return false;

float t = dot((plane.pos-origin), plane.normal)/dotProd;

if(t < ray.t && t >0 )
{
   ray.t = t;
   intersec.normal = plane.normal;
   intersec.pos = origin + t*ray.dir;
   intersec.w_out = origin-intersec.pos;
   intersec.w_out = intersec.w_out / norm(intersec.w_out);
   return true;
}
return false;
}
__device__
void
generateRay(int i_row, int i_col, DeviceRay& ray, DeviceImage& img){

    float pix_size_row = 1.0 / (float)(img.n_rows);
    float pix_size_col = 1.0 / (float)(img.n_cols);

    // assume [-0.5,0.5] x [-0.5,0.5] image plane with 1.0 focal length
    float y = (float)(i_row - img.n_rows/2) * pix_size_row + pix_size_row / 2.0;
    float x = (float)(i_col - img.n_cols/2) * pix_size_col + pix_size_col / 2.0;
    float z = -1.0;

    ray.dir = vec3( x, y, z);
    ray.dir /= norm(ray.dir);
    ray.t = std::numeric_limits<float>::max();
}
__device__
bool
traceRay( DeviceRay& ray, DeviceIntersection& intersec, DeviceScene& scene) {

  bool has_hit = false;
  for (int i = 0; i < scene.numPlanes; ++i) {
       has_hit |= intersect_plane(ray,intersec,scene.planes[i]);
}
  for (int i = 0; i < scene.numSpheres; ++i) {
       has_hit |= intersect_sphere(ray,intersec,scene.spheres[i]);
}

  return has_hit;
}


__device__
float
shade( DeviceIntersection& intersec, DeviceScene scene) {

float shade = 0.0;
    for( unsigned int i = 0; i < scene.numLights; ++i) {
        DevicePLight light = scene.lights[i];
        vec3 w_in = light.pos - intersec.pos;
        w_in /= norm( w_in);
        float matshade = 0;
        vec3 normal = intersec.normal/norm(intersec.normal); //is already normalized
        vec3 r = -w_in+dot(w_in,normal)*normal; //R is already normalized

        if(dot(normal,w_in) >= 0)
            matshade = dot(normal,w_in)  *(scene.material.kd+scene.material.ks*pow(dot(r,intersec.w_out),scene.material.alpha));
        float nshade = light.ill * matshade;
      shade += nshade;
    }

  return shade;
}


////////////////////////////////////////////////////////////////////////////////////////////////////
//! Entry point to device
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__
void
render(DeviceData device_data) {

    DeviceIntersection intersec;
    DeviceRay ray;

    unsigned int rayX = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int rayY = threadIdx.y + blockIdx.y * blockDim.y;

    if(rayX < device_data.img.n_cols && rayY < device_data.img.n_rows){
        generateRay(rayX, rayY, ray,device_data.img);

    //    device_data.img.img[rayX+device_data.img.n_cols*rayY] = 255.0*abs(ray.dir.y);
        if(traceRay(ray,intersec,device_data.scene))
            device_data.img.img[rayY+device_data.img.n_cols*rayX] = shade(intersec,device_data.scene);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//! Initialize device
////////////////////////////////////////////////////////////////////////////////////////////////////
bool
initDevice() {

    int deviceCount = 0;
      checkErrorsCuda( cudaGetDeviceCount(&deviceCount));

      if( 0 == deviceCount) {
        std::cerr << "initDevice() : No CUDA device found." << std::endl;
        return false;
      }

      checkErrorsCuda( cudaSetDevice(0));
      cudaDeviceProp device_props;

          checkErrorsCuda( cudaGetDeviceProperties(&device_props, 0));
          max_threads_per_block = device_props.maxThreadsPerBlock;

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//! Initialize memory
////////////////////////////////////////////////////////////////////////////////////////////////////
bool
initDeviceMemory( const Scene& scene, const Image& image) {

    data.img.n_cols = image.n_cols;
    data.img.n_rows = image.n_rows;

    std::vector<Sphere*> spheres;
    std::vector<Plane*> planes;
    for(Geometry* object : scene.geometry){

        bool isSphere = dynamic_cast<Sphere*>(object);
        if(isSphere)
        {
            spheres.push_back(dynamic_cast<Sphere*>(object));
        }
        else
        {
            planes.push_back(dynamic_cast<Plane*>(object));
        }

    }
    std::vector<DeviceSphere> dspheres;
    std::vector<DevicePlane> dplanes;
    std::vector<DevicePLight> dplights;
for(Sphere* obj : spheres)
{
    DeviceSphere sph;
    sph.center = obj->pos;
    sph.r = obj->r;
    dspheres.push_back(sph);
}
for(Plane* obj : planes)
{
    DevicePlane pln;
    pln.pos = obj->pos;
    pln.normal = obj->n;
    pln.t1 = obj->t1;
    pln.t2 = obj->t2;
    pln.size = obj->size;
    dplanes.push_back(pln);
}
    data.scene.numSpheres = dspheres.size();
    data.scene.numPlanes = dplanes.size();

    for(PointLight* pl : scene.lights)
    {
        DevicePLight dpl;
        dpl.ill = pl->ell;
        dpl.pos = pl->pos;
        dplights.push_back(dpl);
    }
    data.scene.numLights = dplights.size();

    data.scene.material.alpha = 32.0;
    data.scene.material.kd = 1.0;
    data.scene.material.ks = 0.5;

    // allocate device memory
    checkErrorsCuda( cudaMalloc((void **) &(data.scene.spheres), sizeof(DeviceSphere) * dspheres.size()));
    checkErrorsCuda( cudaMalloc((void **) &(data.scene.planes), sizeof(DevicePlane) * dplanes.size()));
  checkErrorsCuda( cudaMalloc((void **) &(data.scene.lights), sizeof(DevicePLight) * dplights.size()));
 checkErrorsCuda( cudaMalloc((void **) &(data.img.img), sizeof(float) * data.img.n_cols *data.img.n_rows));
    // copy device memory

     checkErrorsCuda( cudaMemcpy( data.img.img, image.data.data(), sizeof(float) * image.data.size(), cudaMemcpyHostToDevice));
    checkErrorsCuda( cudaMemcpy( (data.scene.spheres), dspheres.data(), sizeof(DeviceSphere) * dspheres.size(), cudaMemcpyHostToDevice));
     checkErrorsCuda( cudaMemcpy( (data.scene.lights), (dplights.data()), sizeof(DevicePLight) * dplights.size(), cudaMemcpyHostToDevice));
      checkErrorsCuda( cudaMemcpy( (data.scene.planes), (dplanes.data()), sizeof(DevicePlane) * dplanes.size(), cudaMemcpyHostToDevice));

    return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//! Initialize device
////////////////////////////////////////////////////////////////////////////////////////////////////
bool
runDevice( const unsigned int n_rows, const unsigned int n_cols) {
    dim3 num_threads_per_block;
    dim3 num_blocks;

    int block_size = std::sqrt(max_threads_per_block); //1024->32

    num_threads_per_block.x = static_cast<unsigned int>(block_size);
    num_threads_per_block.y = static_cast<unsigned int>(block_size);

     num_blocks.x = std::ceil(n_cols/num_threads_per_block.x);
      num_blocks.y = std::ceil(n_rows/num_threads_per_block.y);

    std::cout << "num_blocks = " << num_blocks.x << " / " << num_blocks.y << std::endl;
    std::cout << "num_threads_per_block = " << num_threads_per_block.x << " / " << num_threads_per_block.y << std::endl;

    render<<< num_blocks, num_threads_per_block >>>(data);

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//! Get image from device
////////////////////////////////////////////////////////////////////////////////////////////////////
bool
getImageDevice( Image& image) {

  checkErrorsCuda(cudaMemcpy( &(image.data[0]), (data.img.img), sizeof(float) * data.img.n_cols*data.img.n_rows, cudaMemcpyDeviceToHost));
  for (int i = 0; i < 20; ++i) {
      printf("%f\n", image.data[i]);
  }
  return true;

}

////////////////////////////////////////////////////////////////////////////////////////////////////
//! Cleanup device
////////////////////////////////////////////////////////////////////////////////////////////////////
void
cleanupDevice() {
     checkErrorsCuda( cudaFree((data.scene.spheres)));
     checkErrorsCuda( cudaFree( (data.scene.planes)));
     checkErrorsCuda( cudaFree((data.scene.lights)));
     checkErrorsCuda( cudaFree( (data.img.img)));
}



