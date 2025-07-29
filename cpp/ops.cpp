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
#include <numeric>
#include <tuple>
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

std::tuple<Matrix, std::vector<double>, Matrix> svd(Matrix A, 
                                                    bool verbose=false) {
  size_t r = std::min(A.nrow, A.ncol);
  Matrix U(A.nrow*r, A.nrow, r);
  Matrix Vt(r*A.ncol, r, A.ncol);
  std::vector<double> s(r);

  // sizes in int for DGESVD
  lapack_int m = A.nrow, n = A.ncol;
  lapack_int lda = m, ldu = m, ldvt = r;
  
  if (verbose) {
    A.print();
    printf("A.nrow, A.ncol, r: %d %d %d\n", A.nrow, A.ncol, r);
    printf("m, n: %d %d\n", m, n);
    printf("lda, ldu, ldvt: %d %d %d\n", lda, ldu, ldvt);
  }

  lapack_int info, lwork;
  double *work, wkopt;

  lwork = -1; // Optimal workspace query
  
  // Workspace query
  dgesvd(
    "S", // JOBU: Option for computing all or part of U. 'S' is the first (m,n) columns. 
    "S", // JOBVT: Option for computing all or part of Vt. 'S' is the first (m,n) rows.
    &m, // M: No. of rows of the input matrix.
    &n, // N: No. of columns of the input matrix.
    A.data_ptr, // A: Input matrix. Contents destroyed during the computation.
    &lda, // LDA: Leading dimension of A.
    s.data(), // S: Array holding the singular values of A.
    U.data_ptr, // U: Matrix holding left singular vectors of A.
    &ldu, // LDU: Leading dimension of U.
    Vt.data_ptr, // Vt: Matrix holding the right singular vectors of A.
    &ldvt, // LDVT: Leading dimension of Vt.
    &wkopt, // WORK: Work array containing uncoverged elements on failure.
    &lwork, // LWORK: Dimension of the array WORK.
    &info // INFO: Exit code.
  );

  lwork = (lapack_int) wkopt;
  work  = (double*) malloc(lwork * sizeof(double));

  if (verbose) {
    printf("Size of the array: %d\n", lwork);
    printf("Exit code for DGESVD: %d\n", info);
  }

  // Compute the thin SVD
  dgesvd(
    "S", // JOBU: Option for computing all or part of U. 'S' is the first (m,n) columns. 
    "S", // JOBVT: Option for computing all or part of Vt. 'S' is the first (m,n) rows.
    &m, // M: No. of rows of the input matrix.
    &n, // N: No. of columns of the input matrix.
    A.data_ptr, // A: Input matrix. Contents destroyed during the computation.
    &lda, // LDA: Leading dimension of A.
    s.data(), // S: Array holding the singular values of A.
    U.data_ptr, // U: Matrix holding left singular vectors of A.
    &ldu, // LDU: Leading dimension of U.
    Vt.data_ptr, // Vt: Matrix holding the right singular vectors of A.
    &ldvt, // LDVT: Leading dimension of Vt.
    work, // WORK: Work array containing uncoverged elements on failure.
    &lwork, // LWORK: Dimension of the array WORK.
    &info // INFO: Exit code.
  );

  if (verbose) {
    printf("Exit code for DGESVD: %d\n", info);
    A.print();
  }

  // Free workspace
  free(work);

  return std::make_tuple(U, s, Vt);
}


Tensor ttm_loop(Tensor& T, Matrix& M, size_t mode){
    std::vector<size_t> ten_dims = T.getdims();
    std::vector<size_t> mat_dims = M.getdims();
    assert(mat_dims[1] == ten_dims[mode]);

    std::vector<size_t> out_ten_dims(ten_dims);
    out_ten_dims[mode] = mat_dims[0];
    size_t buflen = 1;
    for (size_t i = 0; i<out_ten_dims.size(); i++) buflen = buflen * out_ten_dims[i];

    Tensor TO(buflen, out_ten_dims.size(), out_ten_dims);

    if(mode == 0) {
        // TTM on first mode
        size_t cblas_m = mat_dims[0];
        size_t cblas_k = ten_dims[mode]; // Or mat_dims[1]
        size_t cblas_n = 1;
        for (size_t i = mode+1; i<out_ten_dims.size(); i++){
            cblas_n = cblas_n * out_ten_dims[i];
        }
        float cblas_alpha = 1.0;
        float cblas_beta = 0.0;
        auto cblas_a = M.data_ptr;
        auto cblas_lda = mat_dims[0];
        auto cblas_b = T.data_ptr;
        auto cblas_ldb = ten_dims[0];
        auto cblas_c = TO.data_ptr; 
        auto cblas_ldc = out_ten_dims[0]; // Number of rows of the matrix
                                         
        cblas_dgemm(
            CblasColMajor, // Column major order. `Layout` parameter of MKL cblas call.
            CblasNoTrans, // A matrix is not transpose. `transa` param of MKL cblas call.
            CblasNoTrans, // B matrix is not transpose. `transb` param of MKL cblas call.
            cblas_m, // Number of rows of A or C. `m` param of MKL cblas call.
            cblas_n, // Number of cols of B or C. `n` param of MKL cblas call.
            cblas_k, // Inner dimension - number of columns of A or number of rows of B. `k` param of MKL cblas call.
            cblas_alpha, // Scalar `alpha` param of MKL cblas call.
            cblas_a, // Data buffer of A. `a` param of MKL cblas call.
            cblas_lda, // Leading dimension of A. `lda` param of MKL cblas call.
            cblas_b, // Data buffer of B. `b` param of MKL cblas call.
            cblas_ldb, // Leading dimension of B. `ldb` param of MKL cblas call.
            cblas_beta, // Scalar `beta` param of MKL cblas call.
            cblas_c, // Data buffer of C. `c` param of MKL cblas call.
            cblas_ldc // Leading dimension of C. `ldc` param of MKL cblas call.
        );
    }
    //else if(mode == ten_dims.size()-1) {
        //// TTM on last mode
    //}
    else {
        // Variable names are following Algorithm 3.1 from Tensor textbook
        size_t Mk = 1;
        for (size_t i = 0; i<mode; i++){
            Mk = Mk * ten_dims[i];
        }
        size_t Pk = 1;
        for (size_t i = mode+1; i<ten_dims.size(); i++){
            Pk = Pk * ten_dims[i];
        }
        size_t T_stride_len = Mk * ten_dims[mode];
        size_t TO_stride_len = Mk * out_ten_dims[mode];
        for(size_t l = 0; l < Pk; l++){
            size_t cblas_m = Mk;
            size_t cblas_k = ten_dims[mode]; // Or mat_dims[1]
            size_t cblas_n = mat_dims[0];
            float cblas_alpha = 1.0;
            float cblas_beta = 0.0;
            auto cblas_a = T.data_ptr + T_stride_len * l;
            auto cblas_lda = Mk;
            auto cblas_b = M.data_ptr;
            auto cblas_ldb = mat_dims[0];
            auto cblas_c = TO.data_ptr + TO_stride_len * l; 
            auto cblas_ldc = Mk;
          
            cblas_dgemm(
                CblasColMajor, // Column major order. `Layout` parameter of MKL cblas call.
                CblasNoTrans, // A matrix is not transpose. `transa` param of MKL cblas call.
                CblasTrans, // B matrix is transpose. `transb` param of MKL cblas call.
                cblas_m, // Number of rows of A or C. `m` param of MKL cblas call.
                cblas_n, // Number of cols of B or C. `n` param of MKL cblas call.
                cblas_k, // Inner dimension - number of columns of A or number of rows of B. `k` param of MKL cblas call.
                cblas_alpha, // Scalar `alpha` param of MKL cblas call.
                cblas_a, // Data buffer of A. `a` param of MKL cblas call.
                cblas_lda, // Leading dimension of A. `lda` param of MKL cblas call.
                cblas_b, // Data buffer of B. `b` param of MKL cblas call.
                cblas_ldb, // Leading dimension of B. `ldb` param of MKL cblas call.
                cblas_beta, // Scalar `beta` param of MKL cblas call.
                cblas_c, // Data buffer of C. `c` param of MKL cblas call.
                cblas_ldc // Leading dimension of C. `ldc` param of MKL cblas call.
            );
            
        }

    }

    return TO;
}

Tensor ttm(Tensor& T, Matrix& M, size_t mode){
    std::vector<size_t> ten_dims = T.getdims();
    std::vector<size_t> mat_dims = M.getdims();
    assert(mat_dims[1] == ten_dims[mode]);

    std::vector<size_t> out_ten_dims(ten_dims);
    out_ten_dims[mode] = mat_dims[0];
    size_t buflen = 1;
    for (size_t i = 0; i<out_ten_dims.size(); i++) buflen = buflen * out_ten_dims[i];

    Tensor TO(buflen, out_ten_dims.size(), out_ten_dims);

    if(mode == 0) {
        // TTM on first mode
        size_t cblas_m = mat_dims[0];
        size_t cblas_k = ten_dims[mode]; // Or mat_dims[1]
        size_t cblas_n = 1;
        for (size_t i = mode+1; i<out_ten_dims.size(); i++){
            cblas_n = cblas_n * out_ten_dims[i];
        }
        float cblas_alpha = 1.0;
        float cblas_beta = 0.0;
        auto cblas_a = M.data_ptr;
        auto cblas_lda = mat_dims[0];
        auto cblas_b = T.data_ptr;
        auto cblas_ldb = ten_dims[0];
        auto cblas_c = TO.data_ptr; 
        auto cblas_ldc = out_ten_dims[0]; // Number of rows of the matrix
                                         
        cblas_dgemm(
            CblasColMajor, // Column major order. `Layout` parameter of MKL cblas call.
            CblasNoTrans, // A matrix is not transpose. `transa` param of MKL cblas call.
            CblasNoTrans, // B matrix is not transpose. `transb` param of MKL cblas call.
            cblas_m, // Number of rows of A or C. `m` param of MKL cblas call.
            cblas_n, // Number of cols of B or C. `n` param of MKL cblas call.
            cblas_k, // Inner dimension - number of columns of A or number of rows of B. `k` param of MKL cblas call.
            cblas_alpha, // Scalar `alpha` param of MKL cblas call.
            cblas_a, // Data buffer of A. `a` param of MKL cblas call.
            cblas_lda, // Leading dimension of A. `lda` param of MKL cblas call.
            cblas_b, // Data buffer of B. `b` param of MKL cblas call.
            cblas_ldb, // Leading dimension of B. `ldb` param of MKL cblas call.
            cblas_beta, // Scalar `beta` param of MKL cblas call.
            cblas_c, // Data buffer of C. `c` param of MKL cblas call.
            cblas_ldc // Leading dimension of C. `ldc` param of MKL cblas call.
        );
    }
    else {
        // Variable names are following Algorithm 3.1 from Tensor textbook
        size_t Mk = 1;
        for (size_t i = 0; i<mode; i++){
            Mk = Mk * ten_dims[i];
        }
        size_t Pk = 1;
        for (size_t i = mode+1; i<ten_dims.size(); i++){
            Pk = Pk * ten_dims[i];
        }
        //size_t T_stride_len = Mk * ten_dims[mode];
        //size_t TO_stride_len = Mk * out_ten_dims[mode];

        // MKL Strided Batched BLAS documentation: 
        // https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/cblas-gemm-batch-strided.html
        size_t cblas_m = Mk;
        size_t cblas_k = ten_dims[mode]; // Or mat_dims[1]
        size_t cblas_n = mat_dims[0];
        float cblas_alpha = 1.0;
        float cblas_beta = 0.0;
        auto cblas_a = T.data_ptr;
        auto cblas_lda = Mk;
        auto cblas_stridea = Mk * ten_dims[mode];
        auto cblas_b = M.data_ptr;
        auto cblas_ldb = mat_dims[0];
        auto cblas_strideb = 0;
        auto cblas_c = TO.data_ptr; 
        auto cblas_ldc = Mk;
        auto cblas_stridec = Mk * out_ten_dims[mode];
        auto cblas_batch_size = Pk;

        cblas_dgemm_batch_strided(
            CblasColMajor, // Column major order. `Layout` parameter of MKL cblas call.
            CblasNoTrans, // A matrix is not transpose. `transa` param of MKL cblas call.
            CblasTrans, // B matrix is transpose. `transb` param of MKL cblas call.
            cblas_m, // Number of rows of A or C. `m` param of MKL cblas call.
            cblas_n, // Number of cols of B or C. `n` param of MKL cblas call.
            cblas_k, // Inner dimension - number of columns of A or number of rows of B. `k` param of MKL cblas call.
            cblas_alpha, // Scalar `alpha` param of MKL cblas call.
            cblas_a, // Data buffer of A. `a` param of MKL cblas call.
            cblas_lda, // Leading dimension of A. `lda` param of MKL cblas call.
            cblas_stridea,
            cblas_b, // Data buffer of B. `b` param of MKL cblas call.
            cblas_ldb, // Leading dimension of B. `ldb` param of MKL cblas call.
            cblas_strideb,
            cblas_beta, // Scalar `beta` param of MKL cblas call.
            cblas_c, // Data buffer of C. `c` param of MKL cblas call.
            cblas_ldc, // Leading dimension of C. `ldc` param of MKL cblas call.
            cblas_stridec,
            cblas_batch_size
        );

    }

    return TO;
}

std::tuple<Tensor, Matrix, Tensor> slicewise_svd(Tensor A, bool verbose=false) {
  size_t r = std::min(A.dims[0], A.dims[1]);

  // Create the variables
  std::vector<size_t> Udims = A.dims;
  Udims[1] = r;
  size_t Ubuflen = std::accumulate(Udims.begin(), Udims.end(), 1, std::multiplies<size_t>());
  Tensor U(Ubuflen, A.ndim, Udims);

  std::vector<size_t> Vtdims = A.dims;
  Vtdims[0] = r;
  size_t Vtbuflen = std::accumulate(Vtdims.begin(), Vtdims.end(), 1, std::multiplies<size_t>());
  Tensor Vt(Vtbuflen, A.ndim, Vtdims);

  Matrix S(r*A.nslices, r, A.nslices);
  

  // Call slice-wise SVDs
  for (size_t i = 0; i < A.nslices; i++) {
    Matrix Us(Udims[0] * r, Udims[0], r);
    Matrix Vst(r * Vtdims[1], r, Vtdims[1]);
    std::vector<double> s;

    // Compute the SVD
    std::tie(Us, s, Vst) = svd(A.getfrontalslice(i));

    // Set the output tensors
    U.setfrontalslice(Us, i);
    S.setcol(s, i);
    Vt.setfrontalslice(Vst, i);
  }

  return std::make_tuple(U, S, Vt);
}

#endif
