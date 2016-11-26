/// @file
////////////////////////////////////////////////////////////////////////////////////////////////////
///
/// Copyright (C) 2016/17      Christian Lessig, Otto-von-Guericke Universitaet Magdeburg
///
////////////////////////////////////////////////////////////////////////////////////////////////////
///
///  module     : lecture 3
///
///  author     : lessig@isg.cs.ovgu.de
///
///  project    : GPU Programming
///
///  description: race conditions
///
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <vector>
#include <thread>
#include <cstdlib>
#include <cmath>
#include <atomic>



////////////////////////////////////////////////////////////////////////////////////////////////////
// add two vectors together
////////////////////////////////////////////////////////////////////////////////////////////////////
void
compute_norm( unsigned int tid, unsigned int num_threads, int* a, int n, std::atomic<int>& sum) {
//unterteile Vektor in chunks
  // TODO: compute indices for workload for thread
        unsigned int delta = n / num_threads; //n muss ein vielfaches von num_threads sein
      unsigned int i1 = tid * delta;
      unsigned int i2 = (tid+1) * delta;

  // TODO: compute part of norm computation
      for( unsigned int i = i1; i < i2; ++i) {

          sum += a[i] * a[i];
        }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// program entry point
////////////////////////////////////////////////////////////////////////////////////////////////////
int
main( int /*argc*/, char** /*argv*/ ) {

  const unsigned int n = 8192;
  const unsigned int num_threads = 8;

  // TODO: allocate integer array
  int* a = (int*) malloc(n*sizeof(int));
  // TODO: initialize integer array
 for(unsigned int i=0;i<n;i++)
 {
     a[i] = 1;
 }

    std::atomic<int> sum(0);

  // TODO: generate num_threads threads and start parallel execution
  std::vector<std::thread> threads; //threads hat hier noch größe null

    for(unsigned int i = 0; i < num_threads; i++)
    {
        threads.push_back(std::thread( compute_norm, i, 8,a,n,std::ref(sum))); //compute_norm ist function pointer,
        //referenz muss objekt sein, kann man nicht einfach so übergeben
    }
  // TODO: wait for all threads to finish
    for ( auto& thread : threads) {
        thread.join();
    }

  // output
  std::cerr << "norm(a) = " << std::sqrt( (float) sum) << std::endl;

  // TODO: clean up memory
  free(a);

  return EXIT_SUCCESS;
}
