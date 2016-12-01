/// @file
////////////////////////////////////////////////////////////////////////////////////////////////////
///
/// Copyright (C) 2016/17      Christian Lessig, Otto-von-Guericke Universitaet Magdeburg
///
////////////////////////////////////////////////////////////////////////////////////////////////////
///
///  module     : lecture 6
///
///  author     : lessig@isg.cs.ovgu.de
///
///  project    : GPU Programming
///
///  description: reduction in Cuda
///
////////////////////////////////////////////////////////////////////////////////////////////////////

// includes, system
#include <iostream>
#include <vector>
#include <chrono>

typedef std::chrono::time_point<std::chrono::high_resolution_clock> tpoint;

////////////////////////////////////////////////////////////////////////////////////////////////////
// program entry point
////////////////////////////////////////////////////////////////////////////////////////////////////
int
main( int /*argc*/, char** /*argv*/ ) {

  // set up host memory
  // size is chosen so that two reduction steps would suffice
  const unsigned int size = 64 * 1024 * 1024;
  std::vector<float> data( size);
  for( unsigned int i = 0; i < size; ++i) {
    data[i] = 1.0;
  }

  float sum = 0.0;
  tpoint t_start = std::chrono::high_resolution_clock::now();
  for( unsigned int k = 0; k < 1024; ++k) {
    sum = 0.0;
    for( unsigned int i = 0; i < size; ++i) {
      sum += data[i];
    }
  }
  tpoint t_end = std::chrono::high_resolution_clock::now();

  std::cerr << "Result : " << sum << std::endl;

  double wall_clock = std::chrono::duration<double, std::milli>(t_end-t_start).count();
  std::cerr << "Execution time: " <<  wall_clock << " ms."<< std::endl;

  return EXIT_SUCCESS;
}
