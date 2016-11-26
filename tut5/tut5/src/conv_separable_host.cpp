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

// incldues, system
#include <iostream>

// includes, project
#include "kernel_separable.h"
#include "image.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
// convolution on host
////////////////////////////////////////////////////////////////////////////////////////////////////
void
convSeparableHost( float* kdata, const int& kernel_supp_half, const Image& image, Image& image_conv) {

  if( 2 == kernel_supp_half) {

    KernelSeperable<2> kernel( kdata);

    for( int row = 0; row < image.n_rows; ++row) {
      for( int col = 0; col < image.n_cols; ++col) {
        kernel.apply( row, col, &image.data[0], &image_conv.data[0], image.n_rows);
      }
    }
  }
  else {
    std::cerr << "convSeparableHost() :: kernel size not implemented." << std::endl;
  }

}
