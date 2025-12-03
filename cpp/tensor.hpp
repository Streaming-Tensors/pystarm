// tensor.hpp
// Tensor class implementation

#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "utils.hpp"
#include "matrix.hpp"

namespace py = pybind11;

class Tensor{
    public:
    size_t ndim;
    std::vector<size_t> dims;
    size_t nslices;
    double* data_ptr = nullptr; // Expects the data to be in column major order

    // Takes a preallocated buffer by Python
    // py::buffer instead of py:array_t to prevent any chance of silent data copy
    Tensor(py::buffer &buf, size_t ndim, std::vector<size_t> dims){
      #ifdef _MEMPRINT
      std::cout << "Tensor python constructor" << std::endl;
      std::cout << "Pointing to before owning buffer: " << this->data_ptr << std::endl;
      #endif
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
        
        // Following few lines are to test whether memory is contiguous or not
        //size_t n_elem = 1;
        //for(size_t i = 0; i < ndim; i++){
            //n_elem = n_elem * dims[i];
        //}
        
        //double * temp = new double[n_elem];
        //for(size_t i = 0; i < n_elem; i++){
            //printf("%lld\n", i);
            //temp[i] = this->data_ptr[i];
        //}
        
      #ifdef _MEMPRINT
      std::cout << "Pointing to after owning buffer: " << this->data_ptr << std::endl;
      #endif

    }

    // Constructor that allocates new buffer
    Tensor(size_t buflen, size_t ndim, std::vector<size_t> dims){
      #ifdef _MEMPRINT
      std::cout << "Tensor Malloc constructor" << std::endl;
      std::cout << "Pointing to before malloc: " << this->data_ptr << std::endl;
      #endif

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
      #ifdef _MEMPRINT
      std::cout << "Pointing to after malloc: " << this->data_ptr << std::endl;
      #endif
    }

    // Copy constructor (deep copy of date)
    Tensor(const Tensor &obj) : ndim(obj.ndim), dims(obj.dims), nslices(obj.nslices) {
      #ifdef _MEMPRINT
      std::cout << "Tensor Copy constructor" << std::endl;
      std::cout << "Pointing to before copy: " << this->data_ptr << std::endl;
      std::cout << "Copying from: " << obj.data_ptr << std::endl;
      #endif
      size_t buflen = 1;
      for (size_t i = 0; i < ndim; i++) {
        buflen = buflen * dims[i];
      }
      this->data_ptr = (double *) malloc(buflen * sizeof(double));
      std::copy(obj.data_ptr, obj.data_ptr + buflen, this->data_ptr);

      #ifdef _MEMPRINT
      std::cout << "Pointing to after copy: " << this->data_ptr << std::endl;
      #endif
    }

    // Copy assignment operator
    Tensor& operator=(const Tensor &obj) {
      #ifdef _MEMPRINT
      std::cout << "Tensor Copy assignment" << std::endl;
      std::cout << "Pointing to before copy: " << this->data_ptr << std::endl;
      std::cout << "Copying from: " << obj.data_ptr << std::endl;
      #endif

      if (this != &obj) {
        // Free current resource
        this->clear();

        // Copy the other object over
        this->dims    = obj.dims;
        this->ndim    = obj.ndim;
        this->nslices = obj.nslices;

        size_t buflen  = 1;
        for (size_t i = 0; i < ndim; i++) {
          buflen = buflen * dims[i];
        }

        this->data_ptr = (double*) malloc(buflen * sizeof(double));
        std::copy(obj.data_ptr, obj.data_ptr + buflen, this->data_ptr);
      }

      #ifdef _MEMPRINT
      std::cout << "Pointing to after copy: " << this->data_ptr << std::endl;
      #endif

      return *this;
    }
    
    // Move constructor (shallow copy)
    Tensor(Tensor &&obj) noexcept {
      #ifdef _MEMPRINT
      std::cout << "Tensor Move constructor" << std::endl;
      std::cout << "Pointing to before move: " << this->data_ptr << std::endl;
      std::cout << "Moving from: " << obj.data_ptr << std::endl;
      #endif
     
      // Point to the other object 
      this->dims     = std::move(obj.dims);
      this->ndim     = obj.ndim;
      this->nslices  = obj.nslices;
      this->data_ptr = obj.data_ptr;

      // Clear the other other object
      obj.ndim     = 0;
      obj.nslices  = 0;
      obj.data_ptr = nullptr;

      #ifdef _MEMPRINT
      std::cout << "Pointing to after move: " << this->data_ptr << std::endl;
      #endif
    }

    // Move assignment operator
    Tensor& operator=(Tensor &&obj) noexcept {
      #ifdef _MEMPRINT
      std::cout << "Tensor Move assignment" << std::endl;
      std::cout << "Pointing to before move: " << this->data_ptr << std::endl;
      std::cout << "Moving from: " << obj.data_ptr << std::endl;
      #endif

      if (this != &obj) {
        // Free current resource
        this->clear();

        // Point to the other object 
        this->dims     = std::move(obj.dims);
        this->ndim     = obj.ndim;
        this->nslices  = obj.nslices;
        this->data_ptr = obj.data_ptr;

        // Clear the other other object
        obj.ndim     = 0;
        obj.nslices  = 0;
        obj.data_ptr = nullptr;
      }

      #ifdef _MEMPRINT
      std::cout << "Pointing to after move: " << this->data_ptr << std::endl;
      #endif

      return *this;
    }

    // Get tensor dimensions
    std::vector<size_t> getdims(){
        return this->dims;
    }

    /* Get a deep copy of a frontal slice
       Assuming natural ordering
       Slices are linearly indexed (0-based indexing)
    */
    Matrix getfrontalslice_copy(size_t i) const {
      size_t buflen  = this->dims[0] * this->dims[1];
      
      // Check if requesting a legal slice
      assert(i < this->nslices);
  
      size_t start_idx = i * buflen;   
      
    
      Matrix slice_mat(buflen, this->dims[0], this->dims[1]);
      std::copy(this->data_ptr + start_idx, 
        this->data_ptr + (start_idx + buflen), slice_mat.data_ptr);
      
      #ifdef _MEMPRINT
      std::cout << "Tensor memory location: " << this->data_ptr
                << std::endl
                << "Data memory location: " << this->data_ptr + start_idx
                << std::endl << "Matrix memory location: "
                << slice_mat.data_ptr
                << std::endl;
      #endif

      return slice_mat;
    }

    /* Get a frontal slice
       Assuming natural ordering
       Slices are linearly indexed (0-based indexing)
    */
    Matrix getfrontalslice(size_t i) {
      size_t buflen  = this->dims[0] * this->dims[1];
      
      // Check if requesting a legal slice
      assert(i < this->nslices);
  
      size_t start_idx = i * buflen;
    
      Matrix slice_mat(this->data_ptr + start_idx, 
                          this->dims[0], this->dims[1]);

      #ifdef _MEMPRINT
      std::cout << "Tensor memory location: " << this->data_ptr
                << std::endl
                << "Data memory location: " << this->data_ptr + start_idx
                << std::endl << "Matrix memory location: "
                << slice_mat.data_ptr
                << std::endl;
      #endif

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
        #ifdef _MEMPRINT
        std::cout << "Clearing tensor" << std::endl;
        std::cout << "Pointing to: " << this->data_ptr << std::endl;
        #endif
        if (this->data_ptr != nullptr){
            free(this->data_ptr);
        }
        this->data_ptr = nullptr;
    }
};

#endif
