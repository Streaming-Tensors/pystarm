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
#include <cassert>
#include <omp.h>
#include <mkl.h>
#include "matrix.cpp"
#include "tensor.cpp"
#include "ops.cpp"

namespace py = pybind11;

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
        .def("getrow", &Matrix::getrow, "Get i-th row of the matrix")
        .def("setrow", &Matrix::setrow, "Set i-th row of the matrix")
        .def("getcol", &Matrix::getcol, "Get j-th column of the matrix")
        .def("setcol", &Matrix::setcol, "Set j-th column of the matrix")
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
        .def("getdims", &Tensor::getdims, "Get tensor dimensions")
        .def("getfrontalslice", &Tensor::getfrontalslice, "Get frontal slice")
        .def("setfrontalslice", &Tensor::setfrontalslice, "Set frontal slice");
	m.def("matmul", &matmul, "Multiply two matrices and return a new result matrix");
	m.def("svd", &svd, "Compute the thin SVD of a matrix and return a tuple of the factors.",
        py::arg("A"), py::arg("verbose") = false);
	m.def("ttm_loop", &ttm_loop, "Tensor times matrix multiply on a specific mode by looping");
	m.def("ttm", &ttm, "Tensor times matrix multiply on a specific mode by batched BLAS");
	m.def("slicewise_svd", &slicewise_svd, "Compute the slice-wise thin SVD of a tensor",
        py::arg("A"), py::arg("verbose") = false);
}

#endif
