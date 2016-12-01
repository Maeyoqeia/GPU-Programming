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
///  description: Image
///
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _UTIL_H_
#define _UTIL_H_

// namespace, unnamed
namespace Util {

template <class Vec3 >
float
norm2( const Vec3& vec) {
  return vec.x*vec.x + vec.y*vec.y + vec.z*vec.z;
}

template <class Vec3 >
float
norm( const Vec3& vec) {
  return std::sqrt( norm2(vec));
}

template <class Vec3 >
float
dot( const Vec3& vec1, const Vec3& vec2) {
  return vec1.x*vec2.x + vec1.y*vec2.y + vec1.z*vec2.z;
}

}
#endif // _GEOMETRY_H_
