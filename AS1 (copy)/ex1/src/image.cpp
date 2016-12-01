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
///  description: image
///
////////////////////////////////////////////////////////////////////////////////////////////////////

// includes, project
#include "image.h"

// includes, system
#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <string>
////////////////////////////////////////////////////////////////////////////////////////////////////
// constructor, user
////////////////////////////////////////////////////////////////////////////////////////////////////
Image::Image( int num_rows, int num_cols) :
  n_rows( num_rows),
  n_cols( num_cols),
  data( num_rows * num_cols, 0)
{ }

////////////////////////////////////////////////////////////////////////////////////////////////////
// set pixel
////////////////////////////////////////////////////////////////////////////////////////////////////
const float&
Image::operator()( const int& i_row, const int& i_col) const {

  assert( i_row < n_rows);
  assert( i_col < n_cols);

  return data[i_row * n_cols + i_col];
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// set pixel
////////////////////////////////////////////////////////////////////////////////////////////////////
float&
Image::operator()( const int& i_row, const int& i_col) {

  assert( i_row < n_rows);
  assert( i_col < n_cols);

  return data[i_row * n_cols + i_col];
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// set pixel block
////////////////////////////////////////////////////////////////////////////////////////////////////
void
Image::setBlock( int rows_start, int rows_end, int cols_start, int cols_end, const Image& im) {

  int i_row_src = 0;
  int i_col_src = 0;
  for( int i_row = rows_start; i_row < rows_end; ++i_row) {
    i_col_src = 0;
    for( int i_col = cols_start; i_col < cols_end; ++i_col) {
      data[i_row * n_cols + i_col] = im( i_row_src, i_col_src);
      ++i_col_src;
    }
    ++i_row_src;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// constructor, user
////////////////////////////////////////////////////////////////////////////////////////////////////
void
Image::write( const std::string& fname) const {

  // open file and check
  std::fstream file( fname, std::ios::out);
  if( ! file.good()) {
    std::cerr << "Image::write() : Failed to open \"" << fname << "\"" << std::endl;
    return;
  }

  // write header
  file << "P2\n";
  file << "# Simple example image\n";
  file << n_rows << " " << n_cols << '\n';
  file << 255 << '\n';

  // write image data (y axis is flipped in image format)
  for( int i_row = n_rows-1; i_row >= 0; --i_row) {
    for( int i_col = 0; i_col < n_cols; ++i_col) {
      // clamp if necessary
          file << std::min( 255, static_cast<int>( std::round( data[i_row * n_cols + i_col]))) << " ";

    }
    file << '\n';
  }


  if( ! file.good()) {
    std::cerr << "Image::write() : Failed to write '" << fname << "''" << std::endl;
    return;
  }
  file.close();
}
