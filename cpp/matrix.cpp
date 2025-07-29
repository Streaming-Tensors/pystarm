// matrix.cpp
// Matrix class implementation

#ifndef MATRIX_CPP
#define MATRIX_CPP

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

/*
 * pybind11 memory management resources:
 * - Prevent automatic data copy
 *      - https://stackoverflow.com/questions/54793539/pybind11-modify-numpy-array-from-c
 *      - https://stackoverflow.com/questions/67663847/disallow-copy-with-functions-taking-pybind11-arrays
 *      - https://medium.com/@ahmedfgad/handling-python-numpy-arrays-in-c-using-pybind11-0b7450f4f4b3
 * - Returning C++ memory as Python buffer
 *      - https://alexsm.com/pybind11-buffer-protocol-opencv-to-numpy/
 *
 * */

class Matrix {
public:
    size_t nrow;
    size_t ncol;
    double* data_ptr; // Expects the data to be in column major order
    
    // Takes a preallocated buffer by Python
    // py::buffer instead of py:array_t to prevent any chance of silent data copy
    Matrix(py::buffer &buf, size_t m, size_t n)
        : nrow(m), ncol(n) {
        py::buffer_info buf_info = buf.request();
        data_ptr = static_cast<double*>(buf_info.ptr);
        //size_t size = buf_info.size; // Total number of elements
        //size_t itemsize = buf_info.itemsize; //Byte-size of each element
        //std::string format = buf_info.format; // Datatype of each element
        //std::vector<long int> shape = buf_info.shape;
        //std::vector<long int> strides = buf_info.strides;

		//// Example: Print buffer information
		//std::cout << "Buffer Info:" << std::endl;
		//std::cout << "  Size: " << size << std::endl;
		//std::cout << "  Item size: " << itemsize << std::endl;
		//std::cout << "  Format: " << format << std::endl;
		//std::cout << "  Shape: ";
		//for (auto dim : shape) {
			//std::cout << dim << " ";
		//}
		//std::cout << std::endl;
		//std::cout << "  Strides: ";
		//for (auto stride : strides) {
			//std::cout << stride << " ";
		//}
		//std::cout << std::endl;
    }
    
    // Constructor that allocates new buffer
    Matrix(size_t buflen, size_t m, size_t n)
        : nrow(m), ncol(n) {
        this->data_ptr = (double*) malloc(buflen * sizeof(double) );
    }
    
    // Constructor that takes a preallocated buffer
    Matrix(double* ptr, size_t m, size_t n)
        : nrow(m), ncol(n) {
        this->data_ptr = static_cast<double*>(ptr);
    }

    // Copy constructor (deep copy of data)
    Matrix(const Matrix &obj) : nrow(obj.nrow), ncol(obj.ncol) {
      size_t buflen  = this->nrow * this->ncol;
      this->data_ptr = (double*) malloc(buflen * sizeof(double));
      std::copy(obj.data_ptr, obj.data_ptr + buflen, this->data_ptr);
    }
    
    ~Matrix() {
        //// Commenting this out to avoid double free
        //std::cout << "Matrix desctructor" << std::endl;
        //if (this->data_ptr != NULL){
            //free(this->data_ptr);
        //}
    }
    
    double get(size_t i, size_t j){
        return *(this->data_ptr + i + j * this->nrow);
    }

    void set(size_t i, size_t j, double val){
        *(this->data_ptr + i + j * this->nrow) = val;
    }

    std::vector<double> getrow(size_t i) {
      assert(i < this->nrow);

      std::vector<double> row;
      
      for (size_t j = 0; j < this->ncol; j++) {
        row.push_back(get(i, j));
      }

      return row;
    }
    
    void setrow(const std::vector<double> &row, size_t i) {
      assert(row.size() == this->ncol);
      assert(i < this->nrow);

      for (size_t j = 0; j < this->ncol; j++) {
        set(i, j, row[j]);
      }
    }

    std::vector<double> getcol(size_t j) {
      assert(j < this->ncol);

      std::vector<double> col;
      
      for (size_t i = 0; i < this->nrow; i++) {
        col.push_back(get(i, j));
      }

      return col;
    }

    void setcol(const std::vector<double> &col, size_t j) {
      assert(col.size() == this->nrow);
      assert(j < this->ncol);

      for (size_t i = 0; i < this->nrow; i++) {
        set(i, j, col[i]);
      }
    }

    void clear(){
        std::cout << "Clearing matrix" << std::endl;
        if (this->data_ptr != NULL){
            free(this->data_ptr);
        }
    }

    std::vector<size_t> getdims(){
        std::vector<size_t> dims(2);
        dims[0] = this->nrow;
        dims[1] = this->ncol;
        return dims;
    }

    void print(){
        for (size_t i = 0; i < this->nrow; ++i) {
            for (size_t j = 0; j < this->ncol; ++j) {
                printf("%.2lf\t", this->get(i, j) );
            }
            printf("\n");
        }  
//#pragma omp parallel
        //for (size_t i = 0; i < this->nrow; ++i) {
            //int t = omp_get_thread_num();
            //printf("Hello from %d\n", t);
        //}  

    }
};

#endif
