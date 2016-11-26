/// @file
////////////////////////////////////////////////////////////////////////////////////////////////////
///
/// Copyright (C) 2016/17      Christian Lessig, Otto-von-Guericke Universitaet Magdeburg
///
////////////////////////////////////////////////////////////////////////////////////////////////////
///
///  module     : tutorial 5
///
///  author     : lessig@isg.cs.ovgu.de
///
///  project    : GPU Programming
///
///  description: CUDA convolution
///
////////////////////////////////////////////////////////////////////////////////////////////////////

// includes, system
#include <cassert>

// includes, project
#include "image.h"

template< int SuppHalf >
class KernelSeperable {

public:

  //! Constructor, user
  KernelSeperable( float* kdata)
  {
    float weight_total = 0.0;
    for( int i = 0; i < 2*SuppHalf+1; ++i) {
      kernel[i] = kdata[i];
      weight_total += kernel[i];
    }

    assert( (weight_total - 1.0) <= std::numeric_limits<float>::epsilon());
  }

  //! Destructor
  ~KernelSeperable() {  }

public:

  void
  apply( const int& row, const int& col,
         const float* image, float* image_conv,
         const int& image_size)
  {
    float weight_row = 0.0;
    float weight = 0.0;

    int ik = 0;
    int jk = 0;

    image_conv[row * image_size + col] = 0.0;

    for( int i = row - SuppHalf; i <= row + SuppHalf; ++i, ++ik) {
      weight_row = kernel[ik];
      jk = 0;
      for( int j = col - SuppHalf; j <= col + SuppHalf; ++j, ++jk) {

        if( ( i < 0 || j < 0) || (i >= image_size) || (j >= image_size)) {
          continue;
        }

        weight = weight_row * kernel[jk];
        image_conv[row * image_size + col] += weight * image[i * image_size + j];
      }
    }

  }

public:

  float kernel[2*SuppHalf + 1];

private:

  // constructor, default
  KernelSeperable();

  // constructor, copy
  KernelSeperable( const KernelSeperable& other);

  // assignment operator
  KernelSeperable& operator=( const KernelSeperable& other);

};
