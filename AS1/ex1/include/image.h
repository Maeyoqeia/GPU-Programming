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

#ifndef _IMAGE_H_
#define _IMAGE_H_

// includes, system
#include <vector>
#include <iostream>

// Simple Image description
class Image {

public:

  //! constructor, user
  Image( int num_rows, int num_cols);

  // destructor
  ~Image() { };

public:

  const float& operator()( const int& i_row, const int& i_col) const;

  float& operator()( const int& i_row, const int& i_col);

public:

  void setPixel( int i_row, int i_col, float& val);

  void setBlock( int rows_start, int rows_end, int cols_start, int cols_end, const Image& im);

  void write( const std::string& fname) const;

public:

  const int n_rows;
  const int n_cols;

public:

  std::vector< float > data;

private:

  // constructor, default
  Image();

  // copy constructor
  Image( const Image&);

  // assignment operator
  Image& operator=( const Image&);

};

#endif // _GEOMETRY_H_
