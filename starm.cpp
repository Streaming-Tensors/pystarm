// starm.cpp

#ifndef STARM_CPP
#define STARM_CPP

#include <pybind11/pybind11.h>
//#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstdio>
#include <memory>
#include <cstddef>
#include <iostream>
#include <vector>
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
        for(int i = 0; i < ndim; i++) this->dims[i] = dims[i];
        py::buffer_info buf_info = buf.request();
        this->data_ptr = static_cast<double*>(buf_info.ptr);
    }

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


Matrix matmul(Matrix& A, Matrix& B){
    Matrix C(A.nrow*B.ncol, A.nrow, B.ncol);
    //for(size_t i = 0; i < C.nrow; i++){
        //for(size_t j = 0; j < C.ncol; j++){
            //C.set(i, j, 0);
            //for (size_t k = 0; k < A.ncol; k++){
                //C.set(i, j, C.get(i, j) + A.get(i, k)*B.get(k, j) );
            //}
        //}
    //}

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                C.nrow, C.ncol, A.ncol,
                1.0, A.data_ptr, A.ncol, // A matrix
                B.data_ptr, B.ncol,      // B matrix
                0.0, C.data_ptr, C.ncol); // C matrix

    return C;
}

Tensor ttm(Tensor& T, Matrix& m, size_t mode){
    return T;
}



//PYBIND11_MODULE(pystarm, m) {
    //py::class_<Matrix>(m, "Matrix")
        //.def(py::init<py::buffer& , size_t, size_t>());
	//m.def("print", &print, "Print the contents of the matrix");
//}

PYBIND11_MODULE(pystarm, m) {
    //https://alexsm.com/pybind11-buffer-protocol-opencv-to-numpy/
    py::class_<Matrix>(m, "Matrix", py::buffer_protocol())
        .def_buffer([](Matrix& mat) -> py::buffer_info{
            return py::buffer_info(
                mat.data_ptr,
                sizeof(double),
                py::format_descriptor<double>::format(),
                1, // Always return as 1d array buffer
                { mat.nrow * mat.ncol },
                {
                    //sizeof(double) * mat.ncol,
                    sizeof(double) // Because always returning as 1d array buffer
                }
            );
        })
        .def(py::init<py::buffer& , size_t, size_t>())
        .def("get", &Matrix::get, "Get (i,j) th entry of the matrix")
        .def("set", &Matrix::set, "Set (i,j) th entry of the matrix")
        .def("clear", &Matrix::clear, "Free memory of the underlying data buffer")
        .def("getdims", &Matrix::getdims, "Get matrix dimensions")
        .def("print", &Matrix::print, "Print contents of the matrix");
    //py::class_<Tensor>(m, "Tensor")
        //.def(py::init<py::buffer& , size_t, std::vector<size_t> >());
    py::class_<Tensor>(m, "Tensor", py::buffer_protocol())
        .def_buffer([](Tensor& ten) -> py::buffer_info{
            size_t nval = 1;
            for (size_t i = 0; i < ten.ndim; i++){
                nval = nval * ten.dims[i];
            }
            return py::buffer_info(
                ten.data_ptr,
                sizeof(double),
                py::format_descriptor<double>::format(),
                1, // Always return as 1d array buffer
                { nval },
                {
                    sizeof(double) // Because always returning as 1d array buffer
                }
            );
        })
        .def(py::init<py::buffer& , size_t, std::vector<size_t> >())
        .def("clear", &Tensor::clear, "Free memory of the underlying data buffer")
        .def("getdims", &Tensor::getdims, "Get tensor dimensions");
	m.def("matmul", &matmul, "Multiply two matrices and return a new result matrix");
	m.def("ttm", &ttm, "Tensor times matrix multiply on a specific mode");
}

#endif
