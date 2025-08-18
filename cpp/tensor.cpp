// tensor.cpp
// Tensor class implementation

#ifndef TENSOR_CPP
#define TENSOR_CPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdio>
#include <memory>
#include <cstddef>
#include <iostream>
#include <vector>
#include <cassert>
#include <omp.h>
#include <mkl.h>
#include "matrix.cpp"

namespace py = pybind11;

class Tensor{
    public:
    size_t ndim;
    std::vector<size_t> dims;
    size_t nslices;
    double* data_ptr; // Expects the data to be in column major order

    // Takes a preallocated buffer by Python
    // py::buffer instead of py:array_t to prevent any chance of silent data copy
    Tensor(py::buffer &buf, size_t ndim, std::vector<size_t> dims){
        this->ndim = ndim;
        this->dims.resize(ndim);
        for(size_t i = 0; i < ndim; i++) this->dims[i] = dims[i];
        py::buffer_info buf_info = buf.request();
        this->data_ptr = static_cast<double*>(buf_info.ptr);

        // Count the number of slices
        size_t nslices = 1;
        for (size_t ii = 2; ii < ndim; ii++) {
          nslices = nslices * this->dims[ii];
        }
        this->nslices = (this->ndim > 2)? nslices : 0;
    }

    // Constructor that allocates new buffer
    Tensor(size_t buflen, size_t ndim, std::vector<size_t> dims){
        assert(ndim == dims.size());
        this->ndim = ndim;
        this->dims = dims;

        size_t x = 1;
        for(size_t i = 0; i < dims.size(); i++){
            x = x * dims[i];
        }
        assert(x == buflen);

        this->data_ptr = (double*) malloc(buflen * sizeof(double) );

        // Count the number of slices
        size_t nslices = 1;
        for (size_t ii = 2; ii < ndim; ii++) {
          nslices = nslices * this->dims[ii];
        }
        this->nslices = (this->ndim > 2)? nslices : 0;
    }

    // Copy constructor (deep copy of date)
    Tensor(const Tensor &obj) : ndim(obj.ndim), dims(obj.dims), nslices(obj.nslices) {
      size_t buflen = 1;
      for (size_t i = 0; i < ndim; i++) {
        buflen = buflen * dims[i];
      }
      this->data_ptr = (double *) malloc(buflen * sizeof(double));
      std::copy(obj.data_ptr, obj.data_ptr + buflen, this->data_ptr);
    }

    // Get tensor dimensions
    std::vector<size_t> getdims(){
        return this->dims;
    }

    /* Get frontal slice
       Assuming natural ordering
       Slices are linearly indexed (0-based indexing)
    */
    Matrix getfrontalslice(size_t i) {
      size_t buflen  = this->dims[0] * this->dims[1];

      // Check if requesting a legal slice
      assert(i < this->nslices);

      size_t start_idx = i * buflen;

      Matrix slice_mat(buflen, this->dims[0], this->dims[1]);
      std::copy(this->data_ptr + start_idx,
        this->data_ptr + (start_idx + buflen), slice_mat.data_ptr);

      return slice_mat;
    }

    /* Set frontal slice
       Assuming natural ordering
       Slices are linearly indexed (0-based indexing)
    */
    void setfrontalslice(const Matrix &slice, size_t i) {
      // Check if requesting a legal slice
      assert(i < this->nslices);

      size_t buflen  = this->dims[0] * this->dims[1];
      size_t start_idx = i * buflen;

      // Copy to the slice
      std::copy(slice.data_ptr, slice.data_ptr + buflen, this->data_ptr + start_idx);
    }

    void clear(){
        std::cout << "Clearing tensor" << std::endl;
        if (this->data_ptr != NULL){
            free(this->data_ptr);
        }
    }
};

class JaggedTensor{
    public:
    size_t nslices;
    std::vector<std::tuple<Matrix, std::vector<double>, Matrix>> slices; // Each slice is a tuple of (U, s, V), singular values are stored as a vector

    JaggedTensor() : nslices(0) {}

    void addfrontalslice(const Matrix U, const std::vector<double> s, const Matrix Vt, double tol=0.0) {
        // Check if the slice is valid
        assert(U.ncol == s.size() && Vt.nrow == s.size());
        size_t r = s.size();
        if (tol > 0.0) {
            // Check if the singular values are above the tolerance
            r = 0;
            while(r < s.size() && s[r] >= tol){
                r++;
            }
        }
        Matrix slice_U(U.nrow*r , U.nrow, r);
        Matrix slice_Vt(r*Vt.ncol, r, Vt.ncol);
        std::vector<double> slice_s(r);

        // Copy the data
        std::copy(U.data_ptr, U.data_ptr + U.nrow * r, slice_U.data_ptr);
        std::copy(s.begin(), s.begin() + r, slice_s.begin());
        // Vt is column major, but we only want to copy r elements from each column
        size_t start = 0;
        for (size_t i = 0; i < Vt.ncol; i++) {
            std::copy(Vt.data_ptr + start, Vt.data_ptr + start + r, slice_Vt.data_ptr + i*r);
            start += Vt.nrow;
        }
        // Add slice to list
        this->slices.push_back(std::make_tuple(slice_U, slice_s, slice_Vt));
        this->nslices++;
    }

    std::tuple<Matrix, std::vector<double>, Matrix> getfrontalslice(size_t i) {
        // Check if requesting a legal slice
        assert(i < this->nslices);
        return this->slices[i];
    }
};

#endif
