// ops.cpp
// Tensor and matrix operation implementation

#ifndef OPS_CPP
#define OPS_CPP

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

    cblas_dgemm(
        CblasColMajor, // Column major order. `Layout` parameter of MKL cblas call.
        CblasNoTrans, // A matrix is not transpose. `transa` param of MKL cblas call.
        CblasNoTrans, // B matrix is not transpose. `transb` param of MKL cblas call.
        C.nrow, // Number of rows of A or C. `m` param of MKL cblas call.
        C.ncol, // Number of cols of B or C. `n` param of MKL cblas call.
        A.ncol, // Inner dimension - number of columns of A or number of rows of B. `k` param of MKL cblas call.
        1.0, // Scalar `alpha` param of MKL cblas call.
        A.data_ptr, // Data buffer of A. `a` param of MKL cblas call.
        A.nrow, // Leading dimension of A. `lda` param of MKL cblas call.
        B.data_ptr, // Data buffer of B. `b` param of MKL cblas call.
        B.nrow, // Leading dimension of B. `ldb` param of MKL cblas call.
        0.0, // Scalar `beta` param of MKL cblas call.
        C.data_ptr, // Data buffer of C. `c` param of MKL cblas call.
        C.nrow // Leading dimension of C. `ldc` param of MKL cblas call.
    );

    return C;
}

Tensor ttm(Tensor& T, Matrix& M, size_t mode){
    std::vector<size_t> ten_dims = T.getdims();
    std::vector<size_t> mat_dims = M.getdims();
    assert(mat_dims[1] == ten_dims[mode]);

    //std::vector<size_t> out_ten_dims(ten_dims);
    //out_ten_dims[mode] = mat_dims[0];
    //size_t buflen = 1;
    //for (size_t i = mode+1; i<ten_dims.size(); i++){
        //if(i != mode) buflen = buflen * out_ten_dims[i];
    //}

    //Tensor TO(buflen, out_ten_dims.size(), out_ten_dims);

    //if(mode == 0) {
        //// TTM on first mode
        //size_t cblas_m = mat_dims[0];
        //size_t cblas_k = ten_dims[mode]; // Or mat_dims[1]
        //size_t cblas_n = 1;
        //for (size_t i = mode+1; i<ten_dims.size(); i++){
            //cblas_n = cblas_n * ten_dims[i];
        //}
    //}
    ////else if(mode == ten_dims.size()-1) {
        ////// TTM on last mode
    ////}
    //else {
    //}

    return T;
}

#endif
