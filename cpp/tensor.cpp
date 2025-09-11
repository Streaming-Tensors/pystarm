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
    // This class currently supports jaggedness only in the last mode
    // TODO: Support jaggedness in first mode (Vt is variable in row dim)
    // TODO: Generalize to support jaggedness in any one mode (need to store mode index and update copy logic everywhere)
    // TODO: Generalize to support jaggedness in multiple modes. Use case for this is not clear...
    public:
    bool last_mode_jagged;  // right now, we implicitly assume this is set to true
    size_t ndim;
    size_t nslices;  // Number of slices
    // Whether the last mode is jagged. If false, then first mode is jagged
    std::vector<size_t> dims; // Dimensions of the first ndim - 1 tensor
    // !!! We assume that only the last dimension can vary
    std::vector<size_t> slice_ranks; // Each element is the rank of the corresponding slice
    double *data_ptr;

    JaggedTensor() : ndim(0), nslices(0) {}

    JaggedTensor(py::buffer &buf, size_t ndim, std::vector<size_t> dims, std::vector<size_t> slice_ranks)
        : ndim(ndim), nslices(slice_ranks.size()) {
        py::buffer_info buf_info = buf.request();
        this->dims.resize(ndim);
        this->slice_ranks.resize(this->nslices);
        for(size_t i = 0; i < ndim; i++){
            this->dims[i] = dims[i];
        }
        for (size_t i = 0; i < this->nslices; i++) {
            this->slice_ranks[i] = slice_ranks[i];
        }
        // TODO: do we want this to be a deep copy instead?
        this->data_ptr = static_cast<double*>(buf_info.ptr);
    }

    JaggedTensor(size_t buflen, size_t ndim, std::vector<size_t> dims, std::vector<size_t> slice_ranks)
        : ndim(ndim), dims(dims), nslices(slice_ranks.size()), slice_ranks(slice_ranks) {
        assert(ndim >= 2);

        size_t expected_buflen = 1;
        for (size_t dim_size: dims) {
            expected_buflen *= dim_size;
        }
        size_t nrows = expected_buflen;
        expected_buflen = 0;
        for (size_t r : slice_ranks) {
            expected_buflen += nrows * r;
        }
        assert(buflen == expected_buflen);
        this->nslices = slice_ranks.size();
        this->slice_ranks.resize(this->nslices);
        for (size_t i = 0; i < this->nslices; i++) {
            this->slice_ranks[i] = slice_ranks[i];
        }
        this->data_ptr = static_cast<double*>(malloc(buflen*sizeof(double)));
    }

    JaggedTensor(const JaggedTensor & obj) : ndim(obj.ndim), nslices(obj.nslices) {
        this->dims.resize(ndim);
        this->slice_ranks.resize(this->nslices);
        for(size_t i = 0; i < ndim; i++){
            this->dims[i] = obj.dims[i];
        }
        for (size_t i = 0; i < this->nslices; i++) {
            this->slice_ranks[i] = obj.slice_ranks[i];
        }
        size_t nrows = 1;
        for (size_t i = 0; i < ndim - 1; i++) {
            nrows = nrows * dims[i];
        }
        size_t buflen = 0;
        for (size_t r : slice_ranks) {
            buflen += nrows * r;
        }
        this->data_ptr = static_cast<double*>(malloc(buflen * sizeof(double)));
        std::copy(obj.data_ptr, obj.data_ptr + buflen, this->data_ptr);
    }

    // Get tensor dimensions
    std::tuple<std::vector<size_t>, std::vector<size_t>> getdims(){
        // last dimension can vary, so return a tuple (dims, slice ranks)
        return std::make_tuple(this->dims, this->slice_ranks);
    }

    Matrix getfrontalslice(size_t i) {
        // Check if requesting a legal slice
        assert(i < this->nslices);
        size_t nrows = 1;
        for (size_t i = 0; i < this->ndim - 1; i++) {
            nrows *= this->dims[i];
        }
        size_t ncols = this->slice_ranks[i];
        size_t buflen = nrows * ncols;

        size_t start_idx = 0;
        for (size_t s = 0; s < i; s++) {
            start_idx += nrows * this->slice_ranks[s];
        }

        Matrix slice_mat(buflen, nrows, ncols);
        std::copy(this->data_ptr + start_idx,
            this->data_ptr + (start_idx + buflen), slice_mat.data_ptr);

        return slice_mat;
    }

    void setfrontalslice(const Matrix &slice, size_t i) {
        // Check if requesting a legal slice
        assert(i < this->nslices);
        size_t nrows = 1;
        for (size_t i = 0; i < this->ndim - 1; i++) {
            nrows *= this->dims[i];
        }
        size_t ncols = this->slice_ranks[i];
        size_t buflen = nrows * ncols;
        // TODO: permute if first mode is jagged (i.e. this is a Vt slice)
        assert(slice.nrow == nrows);
        assert(slice.ncol == ncols);

        size_t start_idx = 0;
        for (size_t s = 0; s < i; s++) {
            start_idx += nrows * this->slice_ranks[s];
        }

        // Copy to the slice
        std::copy(slice.data_ptr, slice.data_ptr + buflen, this->data_ptr + start_idx);
    }

    void clear(){
        std::cout << "Clearing jagged tensor" << std::endl;
        if (this->data_ptr != nullptr){
            free(this->data_ptr);
        }
        this->dims.clear();
        this->slice_ranks.clear();
    }
};

#endif
