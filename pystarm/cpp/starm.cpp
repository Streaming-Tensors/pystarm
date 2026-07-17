// starm.cpp

#ifndef STARM_CPP
#define STARM_CPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "utils.hpp"
#include "matrix.hpp"
#include "tensor.hpp"
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
        .def(py::init<py::buffer& , size_t, size_t>(), 
             py::keep_alive<1, 2>() // Keep the argument 2 (py::buffer) alive at least as long as 
                                    // the argument 1 (the C++ constructed object) is alive
        )
        .def("get", &Matrix::get, "Get (i,j) th entry of the matrix")
        .def("set", &Matrix::set, "Set (i,j) th entry of the matrix")
        .def("getrow", &Matrix::getrow, "Get i-th row of the matrix")
        .def("setrow", &Matrix::setrow, "Set i-th row of the matrix")
        .def("getcol", &Matrix::getcol, "Get j-th column of the matrix")
        .def("setcol", &Matrix::setcol, "Set j-th column of the matrix")
        .def("clear", &Matrix::clear, "Free memory of the underlying data buffer")
        .def("getdims", &Matrix::getdims, "Get matrix dimensions")
        .def("getbuflen", &Matrix::getbuflen, "Get buffer length")
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
        .def(py::init<py::buffer& , size_t, std::vector<size_t> >(),
             py::keep_alive<1, 2>() // Keep the argument 2 (py::buffer) alive at least as long as 
                                    // the argument 1 (the C++ constructed object) is alive
        )
        .def("clear", &Tensor::clear, "Free memory of the underlying data buffer")
        .def("getdims", &Tensor::getdims, "Get tensor dimensions")
        .def("getbuflen", &Tensor::getbuflen, "Get buffer length")
        .def("getfrontalslice", &Tensor::getfrontalslice, "Get a frontal slice")
        .def("getfrontalslice_copy", &Tensor::getfrontalslice_copy, "Get a deep copy of a frontal slice")
        .def("setfrontalslice", &Tensor::setfrontalslice, "Set a frontal slice")
        .def("norm", &Tensor::norm, "Returns the norm of the tensor")
        .def("generate_random", &Tensor::generate_random, "Randomly generates entries of the tensor");

    py::class_<JaggedMatrix>(m, "JaggedMatrix", py::buffer_protocol())
        .def_buffer([](JaggedMatrix& mat) -> py::buffer_info{
            return py::buffer_info(
                mat.data_ptr,
                sizeof(double),
                py::format_descriptor<double>::format(),
                1, // Always return as 1d array buffer
                { mat.col_offset[mat.ncol] },
                {
                    sizeof(double) // Because always returning as 1d array buffer
                }
            );
        })
        .def(py::init<py::buffer&, std::vector<size_t> >(), 
             py::keep_alive<1, 2>() // Keep the argument 2 (py::buffer) alive at least as long as 
                                    // the argument 1 (the C++ constructed object) is alive
        )
        .def_readonly("ncol", &JaggedMatrix::ncol, "Number of columns in the JaggedMatrix")
        .def_readonly("col_ranks", &JaggedMatrix::col_ranks, "Ranks of each column in the JaggedMatrix")
        .def("getbuflen", &JaggedMatrix::getbuflen, "Get buffer length")
        .def("clear", &JaggedMatrix::clear, "Clear the entire buffer of the jagged matrix")
        .def("get", &JaggedMatrix::get, "Get (i,j) th entry of the JaggedMatrix if it is valid")
        .def("set", &JaggedMatrix::set, "Set (i,j) th entry of the JaggedMatrix if it is valid")
        .def("getcol", &JaggedMatrix::getcol, "Get j-th column of the JaggedMatrix")
        .def("setcol", &JaggedMatrix::setcol, "Set j-th column of the JaggedMatrix");

    py::class_<JaggedTensor>(m, "JaggedTensor", py::buffer_protocol())
        .def_buffer([](JaggedTensor& ten) -> py::buffer_info{
            size_t nval = 0;
            // We assume that the slice ranks are correctly set
            // and that the data_ptr has enough memory allocated
            // to hold all the slices
            for (size_t r : ten.slice_ranks) {
                nval += ten.fixed_dim_size * r;
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
        .def(py::init<py::buffer&, size_t, std::vector<size_t>, bool>(), 
             py::arg("buf"), py::arg("fixed_dim_size"), py::arg("slice_ranks"), py::arg("variable_first_mode") = false,
             py::keep_alive<1, 2>() // Keep the argument 2 (py::buffer) alive at least as long as 
                                    // the argument 1 (the C++ constructed object) is alive
        )
        .def_readonly("nslices", &JaggedTensor::nslices, "Number of slices in the tensor")
        .def_readonly("fixed_dim_size", &JaggedTensor::fixed_dim_size, "Number of modes in the tensor")
        .def_readonly("slice_ranks", &JaggedTensor::slice_ranks, "Ranks of each slice in the last mode")
        //.def_readonly("last_mode_jagged", &JaggedTensor::last_mode_jagged, "  Whether the last mode is jagged. If false, then first mode is jagged") --- IGNORE ---
        .def("setfrontalslice", &JaggedTensor::setfrontalslice, "set a frontal slice in the jagged tensor")
        .def("getfrontalslice", &JaggedTensor::getfrontalslice, "Get a frontal slice from the jagged tensor by index")
        .def("getbuflen", &JaggedTensor::getbuflen, "Get buffer length")
        .def("clear", &JaggedTensor::clear, "Clear all slices in the jagged tensor");

	m.def("matmul", &matmul, "Multiply two matrices and return a new result matrix");
	m.def("svdvals", &svdvals, "Compute the singular values of a matrix.",
        py::arg("A"), py::arg("verbose") = false);
	m.def("svd", &svd, "Compute the thin SVD of a matrix and return a tuple of the factors.",
        py::arg("A"), py::arg("verbose") = false);
	m.def("svdx", &svdx, "Compute the truncated SVD of a matrix and return a tuple of the factors.",
        py::arg("A"), py::arg("k"), py::arg("verbose") = false);
	m.def("threshold", &threshold, "Compute the cut-off singular value given a tolerance to meet.",
        py::arg("A"), py::arg("tol"), py::arg("verbose") = false);
	m.def("thresholds", &thresholds, "Compute the column-wise ranks given a tolerance to meet.",
        py::arg("A"), py::arg("tol"), py::arg("verbose") = false);
	m.def("ttm_loop", &ttm_loop, "Tensor times matrix multiply on a specific mode by looping");
	m.def("ttm", &ttm, "Tensor times matrix multiply on a specific mode by batched BLAS");
	m.def("ttm_parfor", &ttm_parfor, "Tensor times matrix multiply on a specific mode by parallel loop with sequential BLAS per block");
	m.def("slicewise_svdvals", &slicewise_svdvals, "Compute the slice-wise singular values of a tensor",
        py::arg("A"), py::arg("verbose") = false);
	m.def("slicewise_svd", &slicewise_svd, "Compute the slice-wise thin SVD of a tensor",
        py::arg("A"), py::arg("verbose") = false);
	m.def("slicewise_svd_seq", &slicewise_svd_seq,
        "Compute slice-wise thin SVD sequentially — one MKL-threaded dgesvd per slice (benchmark variant)",
        py::arg("A"), py::arg("verbose") = false);
	m.def("slicewise_svdx", &slicewise_svdx, "Compute the truncated slice-wise SVD of a tensor",
        py::arg("A"), py::arg("k"), py::arg("verbose") = false);
	m.def("slicewise_svdks", &slicewise_svdks, "Compute the truncated slice-wise SVD of a tensor with different ranks per frontal slice",
        py::arg("A"), py::arg("ks"), py::arg("verbose") = false);
	m.def("slicewise_matmul", &slicewise_matmul, "Compute the slice-wise multiplication of the output of slicewise_svd - U, VT and S");
	m.def("slicewise_matmulks", &slicewise_matmulks, "Compute the slice-wise multiplication of the output of slicewise_svdks - U, VT and S");
	m.def("transform", &transform, "Transform tensor with (multi)ttm in a specified order");
	m.def("tsvdmi_compress", &tsvdmi_compress, "Compress using TSVDM-I algorithm");
	m.def("tsvdmi_reconstruct", &tsvdmi_reconstruct, "Reconstruct output of TSVDM-I algorithm");
	m.def("tsvdmii_compress", &tsvdmii_compress, "Compress using TSVDM-II algorithm");
}

#endif
