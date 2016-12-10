
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
float f_limit = std::numeric_limits<float>::infinity();

unsigned int max_threads_per_block;
DeviceSphere* spheres;
DevicePlane* planes;
struct phongMaterial{
    float ks;
    float kd;
    float alpha;
};
struct deviceImage{
    float* img;
    unsigned int n_cols;
    unsigned int n_rows;
};
class DeviceRay{
public:
    vec3 dir;
    float t;
    DeviceRay(vec3 dir, float t)
    {
        this->dir = dir;
        this->t = t;
    }
};

class DeviceSphere
{
public:
vec3 center;
float r;
phongMaterial material;
DeviceSphere(vec3 center,float r,phongMaterial material) {
    this->center= center;
    this->r=r;
    this->material = material;
}
};

class DevicePlane
{
public:
vec3 pos;
vec3 normal;
vec3 t1;
vec3 t2;
vec2 size;
phongMaterial material;
DeviceSphere(vec3 pos, vec3 normal,vec3 t1,vec3 t2,vec2 size, phongMaterial material) {
    this->pos=pos;
    this->normal=normal;
    this->t1=t1;
    this->t2 = t2;
    this->size=size;
    this->material=material;
}
};
class DeviceIntersection
{
public:
    vec3 pos;
    vec3 w_out;
    vec3 n;
    phongMaterial mat;
    DeviceIntersection(vec3 pos,vec3 w_out,vec3 n,vec3 mat){
        this->pos= pos;
        this->w_out = w_out;
        this->n = n;
        this->mat = mat;
    }
};
}
__device__
void
generateRay(int i_row, int i_col, Ray& ray){

    float pix_size_row = 1.0 / static_cast<float>(deviceImage.n_rows);
    float pix_size_col = 1.0 / static_cast<float>(deviceImage.n_cols);

    // assume [-0.5,0.5] x [-0.5,0.5] image plane with 1.0 focal length
    float y = static_cast<float>(i_row - deviceImage.n_rows/2) * pix_size_row + pix_size_row / 2.0;
    float x = static_cast<float>(i_col - deviceImage.n_cols/2) * pix_size_col + pix_size_col / 2.0;
    float z = -1.0;

    ray.dir = vec3( x, y, z);
    ray.dir /= vectorNorm(ray.dir);
    ray.t = f_limit;
}

__device__
float
vectorNorm(vec3& vector){
    return std::sqrt(vector.x*vector.x+vector.y*vector.y+vector.z*vector.z);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//! Entry point to device
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__
void
render() {

    Intersection* intersec = new Intersection();
    Ray* ray = new Ray();
    for (unsigned int i = 0; i < num_rows; ++i) {
        for (unsigned int j = 0; j < num_cols; ++j) {

            this->cam->generateRay(i,j,*ray);

            if(this->scene->traceRay(*ray, *intersec))
            {
                this->cam->image(i,j) = shade(*intersec);
            }

        }

    }
    delete(ray);
    delete(intersec);
    DeviceRay* ray = new DeviceRay();

    unsigned int rayX = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int rayY = threadIdx.y + blockIdx.y * blockDim.y;

    if(rayX < deviceImage.n_cols && rayY < deviceImage.n_rows){
        generateRay(rayX, rayY, ray);
    }

    free(ray);
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


    checkErrorsCuda(cudaMalloc((void**)&deviceImage.img, sizeof(float) * image.n_rows * image.n_cols));
    checkErrorsCuda( cudaMemcpy( deviceImage.img, scene.geometry[0], sizeof(int) * scene.geometry.size(), cudaMemcpyHostToDevice));
    checkErrorsCuda( cudaMemcpy( image.data, image.data[0], sizeof(int) * image.data.size(), cudaMemcpyHostToDevice));

    checkErrorsCuda(cudaMalloc((void**)&planes, sizeof(DevicePlane)));
    checkErrorsCuda(cudaMalloc((void**)&spheres, sizeof(DeviceSphere)));

    deviceImage.n_cols = image.n_cols;
    deviceImage.n_rows = image.n_rows;

    for(int j = 0; j < image.n_rows; j++){
        for(int i = 0; i < image.n_cols; i++){
            deviceImage.img[i + j*image.n_cols] = image(i,j);
        }
    }


    for(auto object : scene.geometry){
        isSphere = dynamic_cast<Sphere*>(object);
        if(isSphere) numSpheres++;
        else (*numPlanes)++;
    }

    checkErrorsCuda(cudaMalloc((void**)&geometry, sizeof(float) * 4 * numSpheres + sizeof(float) * 9 * (*numPlanes)));

    *planeIndex = numSpheres*4;


    int sphereDex = 0;
    int planeDex = 0;

    for(Geometry* object : scene.geometry){
        isSphere = dynamic_cast<Sphere*>(object);
        if(isSphere){
            //Sphere: [x, y, z, r]
            geometry[sphereDex] = object->pos.x;
            sphereDex++;
            geometry[sphereDex] = object->pos.y;
            sphereDex++;
            geometry[sphereDex] = object->pos.z;
            sphereDex++;

            geometry[sphereDex] = isSphere->r;
            sphereDex++;
        }else{
            //Plane: [t1.x, t1.y, t1.z, t2.x, t2.y, t2.z, n.x, n.y, n.z]
            Plane* plane = dynamic_cast<Plane*>(object);

            geometry[*planeIndex + planeDex] = plane->t1.x;
            planeDex++;
            geometry[*planeIndex + planeDex] = plane->t1.y;
            planeDex++;
            geometry[*planeIndex + planeDex] = plane->t1.z;
            planeDex++;

            geometry[*planeIndex + planeDex] = plane->t2.x;
            planeDex++;
            geometry[*planeIndex + planeDex] = plane->t2.y;
            planeDex++;
            geometry[*planeIndex + planeDex] = plane->t2.z;
            planeDex++;

            geometry[*planeIndex + planeDex] = plane->n.x;
            planeDex++;
            geometry[*planeIndex + planeDex] = plane->n.y;
            planeDex++;
            geometry[*planeIndex + planeDex] = plane->n.z;
            planeDex++;
        }
    }

    return true;
    // allocate device memory
    checkErrorsCuda( cudaMalloc((void **) &scene.geometry, sizeof(int) * scene.geometry.size()));
  checkErrorsCuda( cudaMalloc((void **) &scene.lights, sizeof(int) * scene.lights.size()));
  checkErrorsCuda( cudaMalloc((void **) &image.data, sizeof(int) * scene.data.size()));

    // copy device memory
    checkErrorsCuda( cudaMemcpy( scene.geometry, scene.geometry[0], sizeof(int) * scene.geometry.size(), cudaMemcpyHostToDevice));
     checkErrorsCuda( cudaMemcpy( scene.lights, scene.lights[0], sizeof(int) * scene.lights.size(), cudaMemcpyHostToDevice));
      checkErrorsCuda( cudaMemcpy( image.data, image.data[0], sizeof(int) * image.data.size(), cudaMemcpyHostToDevice));
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//! Initialize device
////////////////////////////////////////////////////////////////////////////////////////////////////
bool
runDevice( const unsigned int n_rows, const unsigned int n_cols) {
    dim3 num_threads_per_block;
    dim3 num_blocks;

    int block_size = sqrt(max_threads_per_block); //1024->32

    num_threads_per_block.x = static_cast<unsigned int>(block_size);
    num_threads_per_block.y = static_cast<unsigned int>(block_size);

     num_blocks.x = ceil(n_cols/num_threads_per_block.x);
      num_blocks.y = ceil(n_rows/num_threads_per_block.y);

    std::cout << "num_blocks = " << num_blocks.x << " / " << num_blocks.y << std::endl;
    std::cout << "num_threads_per_block = " << num_threads_per_block.x << " / " << num_threads_per_block.y << std::endl;

        //ray: [origin.x, origin.y, origin.z, dir.x, dir.y, dir.z, t]


    render<<< num_blocks, num_threads_per_block >>>(n_rows, n_cols);

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//! Get image from device
////////////////////////////////////////////////////////////////////////////////////////////////////
bool
getImageDevice( Image& image) {

  checkErrorsCuda(cudaMemcpy( &image[0], image, sizeof(int) * image.size(), cudaMemcpyDeviceToHost));
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//! Cleanup device
////////////////////////////////////////////////////////////////////////////////////////////////////
void
cleanupDevice() {
     checkErrorsCuda( cudaFree());
}


