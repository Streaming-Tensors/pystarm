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

namespace py = pybind11;

class Tensor{
    public:
    size_t ndim;
    std::vector<size_t> dims;
    double* data_ptr; // Expects the data to be in column major order

    // Takes a preallocated buffer by Python
    // py::buffer instead of py:array_t to prevent any chance of silent data copy
    Tensor(py::buffer &buf, size_t ndim, std::vector<size_t> dims){
        this->ndim = ndim;
        this->dims.resize(ndim);
        for(size_t i = 0; i < ndim; i++) this->dims[i] = dims[i];
        py::buffer_info buf_info = buf.request();
        this->data_ptr = static_cast<double*>(buf_info.ptr);
    }

    // Constructor that allocates new buffer
    Tensor(size_t buflen, size_t ndim, std::vector<size_t> dims){
        assert(ndim == dims.size());
        this->ndim = ndim;
        for(size_t i = 0; i < ndim; i++) this->dims[i] = dims[i];

        size_t x = 1;
        for(size_t i = 0; i < dims.size(); i++){
            x = dims[i];
        }
        assert(x == buflen);

        this->data_ptr = (double*) malloc(buflen * sizeof(double) );
    }
    
    // Get tensor dimensions
    std::vector<size_t> getdims(){
        return this->dims;
    }

    void clear(){
        std::cout << "Clearing tensor" << std::endl;
        if (this->data_ptr != NULL){
            free(this->data_ptr);
        }
    }
};

#endif
