// matrix.hpp
// Matrix class implementation

#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "utils.hpp"

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
        #ifdef _MEMPRINT
        std::cout << "Reg constructor (pybuffer)" << std::endl;
        #endif
        py::buffer_info buf_info = buf.request();
        data_ptr = static_cast<double*>(buf_info.ptr);
        #ifdef _MEMPRINT
        std::cout << "Pointing to: " << data_ptr << std::endl;
        #endif
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
        #ifdef _MEMPRINT
        std::cout << "Reg constructor (malloc)" << std::endl;
        #endif
        this->data_ptr = (double*) malloc(buflen * sizeof(double) );
        #ifdef _MEMPRINT
        std::cout << "Pointing to: " << data_ptr << std::endl;
        #endif
    }

    // Empty constructor
    Matrix()
        : nrow(0), ncol(0) {
        this->data_ptr = nullptr;
    }
    
    // Constructor that takes a preallocated buffer
    Matrix(double* ptr, size_t m, size_t n)
        : nrow(m), ncol(n) {
        #ifdef _MEMPRINT
        std::cout << "Reg constructor (buffer)" << std::endl;
        std::cout << "Pointing to: " << ptr << std::endl;
        #endif
        this->data_ptr = static_cast<double*>(ptr);
    }

    // Copy constructor (deep copy of data)
    Matrix(const Matrix &obj) : nrow(obj.nrow), ncol(obj.ncol) {
      #ifdef _MEMPRINT
      std::cout << "Copy constructor" << std::endl;
      #endif
      size_t buflen  = this->nrow * this->ncol;
      this->data_ptr = (double*) malloc(buflen * sizeof(double));
      std::copy(obj.data_ptr, obj.data_ptr + buflen, this->data_ptr);
      #ifdef _MEMPRINT
      std::cout << "Copying from: " << obj.data_ptr << std::endl;
      std::cout << "Pointing to: " << this->data_ptr << std::endl;
      #endif
    }

    // Copy assignment operator
    Matrix& operator=(const Matrix &obj) {
      #ifdef _MEMPRINT
      std::cout << "Copy assignment" << std::endl;
      #endif

      if (this != &obj) {
        // Free current resource
        this->clear();

        // Copy the other object over
        this->nrow     = obj.nrow;
        this->ncol     = obj.ncol;

        size_t buflen  = this->nrow * this->ncol;
        this->data_ptr = (double*) malloc(buflen * sizeof(double));
        std::copy(obj.data_ptr, obj.data_ptr + buflen, this->data_ptr);
      }

      #ifdef _MEMPRINT
      std::cout << "Copying from: " << obj.data_ptr << std::endl;
      std::cout << "Pointing to: " << this->data_ptr << std::endl;
      #endif

      return *this;
    }

    // Move constructor (shallow copy)
    Matrix(Matrix &&obj) noexcept {
      #ifdef _MEMPRINT
      std::cout << "Move constructor" << std::endl;
      std::cout << "Pointing to: " << obj.data_ptr << std::endl;
      #endif

      // Point to the other object      
      this->nrow = obj.nrow;
      this->ncol = obj.ncol;
      this->data_ptr = obj.data_ptr;

      // Clear the other other object
      obj.nrow     = 0;
      obj.ncol     = 0;
      obj.data_ptr = nullptr;
    }
    
    // Move assignment operator
    Matrix& operator=(Matrix &&obj) noexcept {
      #ifdef _MEMPRINT
      std::cout << "Move assignment" << std::endl;
      std::cout << "Current memory: " << this->data_ptr << std::endl;
      std::cout << "Pointing to: " << obj.data_ptr << std::endl;
      #endif

      if (this != &obj) {
        // Free current resource
        this->clear();

        // Point to the other object
        this->nrow     = obj.nrow;
        this->ncol     = obj.ncol;
        this->data_ptr = obj.data_ptr;

        // Clear the other object
        obj.nrow     = 0;
        obj.ncol     = 0;
        obj.data_ptr = nullptr;
      }

      return *this;
    }
    
    ~Matrix() {
        // Commenting this out to avoid double free
      #ifdef _MEMPRINT
      std::cout << "Matrix destructor" << std::endl;
      std::cout << "Pointing to: " << this->data_ptr << std::endl;
      #endif
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
        #ifdef _MEMPRINT
        std::cout << "Clearing matrix" << std::endl;
        std::cout << "Pointing to: " << this->data_ptr << std::endl;
        #endif
        if (this->data_ptr != nullptr){
            free(this->data_ptr);
        }
        this->data_ptr = nullptr;
        #ifdef _MEMPRINT
        std::cout << "Pointing to: " << this->data_ptr << std::endl;
        #endif
    }

    std::vector<size_t> getdims(){
        std::vector<size_t> dims(2);
        dims[0] = this->nrow;
        dims[1] = this->ncol;
        return dims;
    }

    void print(){
        #ifdef _MEMPRINT
        printf("Matrix::print()\n");        
        printf("Memory location: %p\n", this->data_ptr);        
        #endif

        for (size_t i = 0; i < this->nrow; ++i) {
            for (size_t j = 0; j < this->ncol; ++j) {
                printf("%.16e\t", this->get(i, j) );
            }
            printf("\n");
        }  

        //for (size_t i = 0; i < 1; ++i) {
            //for (size_t j = 0; j < 10; ++j) {
                //printf("%.16e\t", this->get(i, j) );
            //}
            //printf("\n");
        //}  
        //for (size_t i = this->nrow-1; i < this->nrow; ++i) {
            //for (size_t j = 0; j < 10; ++j) {
                //printf("%.16e\t", this->get(i, j) );
            //}
            //printf("\n");
        //}  
//#pragma omp parallel
        //for (size_t i = 0; i < this->nrow; ++i) {
            //int t = omp_get_thread_num();
            //printf("Hello from %d\n", t);
        //}  

    }

    double norm() const {
        #ifdef _MEMPRINT
        printf("Memory location: %p\n", this->data_ptr);        
        #endif

        size_t buflen = this->nrow * this->ncol;
        //double norm = 0.0;
        //for (size_t i = 0; i < buflen; ++i) {
            //norm += this->data_ptr[i] * this->data_ptr[i];
        //}
        //return std::sqrt(norm);

        MKL_INT cblas_n = (MKL_INT) buflen;
        MKL_INT cblas_incx = (MKL_INT)(1);
        double* cblas_x = this->data_ptr;
        return cblas_dnrm2(cblas_n, cblas_x, cblas_incx);
    }
};

class JaggedMatrix{
    // This class supports jaggedness of a matrix in the columns
    public:
    size_t ncol;  // Number of columns in the matrix
    std::vector<size_t> col_ranks;  // Each element is the rank of the corresponding column
    std::vector<size_t> col_offset;  // Contains offset to data_ptr for each column. Contains 1 more element to store the buffer length. 
    double *data_ptr;

    JaggedMatrix(py::buffer &buf, std::vector<size_t> col_ranks)
        : ncol(col_ranks.size()) {
        py::buffer_info buf_info = buf.request();
        this->col_ranks.resize(this->ncol);
        this->col_offset.resize(this->ncol+1, 0);
        for (size_t i = 0; i < this->ncol; i++) {
            this->col_ranks[i] = col_ranks[i];
        }
        std::partial_sum(col_ranks.begin(), col_ranks.end(), col_offset.begin()+1);
        this->data_ptr = static_cast<double*>(buf_info.ptr);
    }

    JaggedMatrix(size_t buflen, std::vector<size_t> col_ranks)
        : ncol(col_ranks.size()) {
        this->col_ranks.resize(this->ncol);
        this->col_offset.resize(this->ncol+1, 0);
        for (size_t i = 0; i < this->ncol; i++) {
            this->col_ranks[i] = col_ranks[i];
        }
        std::partial_sum(col_ranks.begin(), col_ranks.end(), col_offset.begin()+1);
        size_t expected_buflen = this->col_offset[this->ncol];
        assert(buflen == expected_buflen);
        this->data_ptr = static_cast<double*>(malloc(buflen*sizeof(double)));
    }

    void clear(){
        std::cout << "Clearing jagged matrix" << std::endl;
        if (this->data_ptr != nullptr){
            free(this->data_ptr);
        }
        this->col_ranks.clear();
        this->col_offset.clear();
    }
};

#endif
