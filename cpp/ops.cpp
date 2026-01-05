// ops.cpp
// Tensor and matrix operation implementation

#ifndef OPS_CPP
#define OPS_CPP

#include <numeric>
#include <tuple>
#include "utils.hpp"
#include "matrix.hpp"
#include "tensor.hpp"

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
    printf("A.nrow, A.ncol, r: %zu %zu %zu\n", A.nrow, A.ncol, r);
    printf("m, n: %d %d\n", m, n);
    printf("lda, ldu, ldvt: %d %d %d\n", lda, ldu, ldvt);
  }

  // Workspace options
  lapack_int info, lwork;
  double *work, wkopt;

  lwork = -1; // Optimal workspace query
  
  // Workspace query
  dgesvd(
    "S", // JOBU: Option for computing all or part of U. 'S' is the first (m,n) columns. 
    "S", // JOBVT: Option for computing all or part of Vt. 'S' is the first (m,n) rows.
    &m, // M: No. of rows of the input matrix.
    &n, // N: No. of columns of the input matrix.
    A.data_ptr, // A: Input matrix.
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

  // Free workspace and the copied matrix
  free(work);
  A.clear();

  return std::make_tuple(std::move(U), std::move(s), std::move(Vt));
}

std::vector<double> svdvals(Matrix A, bool verbose=false) {
  size_t r = std::min(A.nrow, A.ncol);
  std::vector<double> s(r);

  // sizes in int for DGESVD
  lapack_int m = A.nrow, n = A.ncol;
  lapack_int lda = m, ldu = m, ldvt = r;
  
  if (verbose) {
    A.print();
    printf("A.nrow, A.ncol, r: %zu %zu %zu\n", A.nrow, A.ncol, r);
    printf("m, n: %d %d\n", m, n);
    printf("lda, ldu, ldvt: %d %d %d\n", lda, ldu, ldvt);
  }

  // Workspace options
  lapack_int info, lwork;
  double *work, wkopt;

  lwork = -1; // Optimal workspace query
  
  // Workspace query
  dgesvd(
    "N", // JOBU: Option for computing all or part of U. 'N' is no columns. 
    "N", // JOBVT: Option for computing all or part of Vt. 'N' is the no rows.
    &m, // M: No. of rows of the input matrix.
    &n, // N: No. of columns of the input matrix.
    A.data_ptr, // A: Input matrix.
    &lda, // LDA: Leading dimension of A.
    s.data(), // S: Array holding the singular values of A.
    NULL, // U: Matrix holding left singular vectors of A.
    &ldu, // LDU: Leading dimension of U.
    NULL, // Vt: Matrix holding the right singular vectors of A.
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
    "N", // JOBU: Option for computing all or part of U. 'S' is no columns. 
    "N", // JOBVT: Option for computing all or part of Vt. 'S' is no rows.
    &m, // M: No. of rows of the input matrix.
    &n, // N: No. of columns of the input matrix.
    A.data_ptr, // A: Input matrix. Contents destroyed during the computation.
    &lda, // LDA: Leading dimension of A.
    s.data(), // S: Array holding the singular values of A.
    NULL, // U: Matrix holding left singular vectors of A.
    &ldu, // LDU: Leading dimension of U.
    NULL, // Vt: Matrix holding the right singular vectors of A.
    &ldvt, // LDVT: Leading dimension of Vt.
    work, // WORK: Work array containing uncoverged elements on failure.
    &lwork, // LWORK: Dimension of the array WORK.
    &info // INFO: Exit code.
  );

  if (verbose) {
    printf("Exit code for DGESVD: %d\n", info);
    A.print();
  }

  // Free workspace and the copied matrix
  free(work);
  A.clear();

  return s;
}

std::tuple<Matrix, std::vector<double>, Matrix> svdx(Matrix A, size_t k,
                                                    bool verbose=false) {
  Matrix U(A.nrow*k, A.nrow, k);
  Matrix Vt(k*A.ncol, k, A.ncol);
  std::vector<double> s(std::min(A.nrow, A.ncol));

  // sizes in int for DGESVDX
  lapack_int m = A.nrow, n = A.ncol;
  lapack_int lda = m, ldu = m, ldvt = k;
  
  if (verbose) {
    A.print();
    printf("A.nrow, A.ncol, k: %zu %zu %zu\n", A.nrow, A.ncol, k);
    printf("m, n: %d %d\n", m, n);
    printf("lda, ldu, ldvt: %d %d %d\n", lda, ldu, ldvt);
  }

  // Workspace options
  lapack_int info, lwork;
  lapack_int *iwork;
  double *work, wkopt;

  lwork = -1; // Optimal workspace query
  iwork = (lapack_int*) malloc((12 * std::min(m, n)) * sizeof(lapack_int));

  // Options for DGESVDX
  double vl = 0.0, vu = 0.0;
  lapack_int il = 1, iu = k, ns;
  
  // Workspace query
  dgesvdx(
    "V", // JOBU: Option for computing all or part of U. 'V' is the columns specified by RANGE. 
    "V", // JOBVT: Option for computing all or part of Vt. 'S' is the rows specified by RANGE.
    "I", // RANGE: Option for computing range of singular values. 'I' is [IL, IU] range.
    &m, // M: No. of rows of the input matrix.
    &n, // N: No. of columns of the input matrix.
    A.data_ptr, // A: Input matrix.
    &lda, // LDA: Leading dimension of A.
    &vl, // VL: Lower bound for singular value to search. Not referenced for RANGE "I".
    &vu, // VU: Upper bound for singular value to search. Not referenced for RANGE "I".
    &il, // IL: Lower index of singular values returned. 1 <= IL <= IU <= min(M, N).
    &iu, // IU: Upper index of singular values returned. 1 <= IL <= IU <= min(M, N).
    &ns, // NS: Number of singular values found. Should be IU - IL + 1.
    s.data(), // S: Array holding the singular values of A.
    U.data_ptr, // U: Matrix holding left singular vectors of A.
    &ldu, // LDU: Leading dimension of U.
    Vt.data_ptr, // Vt: Matrix holding the right singular vectors of A.
    &ldvt, // LDVT: Leading dimension of Vt.
    &wkopt, // WORK: Work array containing size of extra memory needed.
    &lwork, // LWORK: Dimension of the array WORK.
    iwork, // IWORK: Iwork array containing indices of uncoverged elements on failure.
    &info // INFO: Exit code.
  );

  lwork = (lapack_int) wkopt;
  work  = (double*) malloc(lwork * sizeof(double));

  if (verbose) {
    printf("Size of the work array: %d\n", lwork);
    printf("Size of the iwork array: %d\n", 12 * std::min(m, n));
    printf("Exit code for DGESVDX: %d\n", info);
  }

  // Compute the truncated SVD
  dgesvdx(
    "V", // JOBU: Option for computing all or part of U. 'V' is the columns specified by RANGE. 
    "V", // JOBVT: Option for computing all or part of Vt. 'S' is the rows specified by RANGE.
    "I", // RANGE: Option for computing range of singular values. 'I' is [IL, IU] range.
    &m, // M: No. of rows of the input matrix.
    &n, // N: No. of columns of the input matrix.
    A.data_ptr, // A: Input matrix. Contents destroyed during the computation.
    &lda, // LDA: Leading dimension of A.
    &vl, // VL: Lower bound for singular value to search. Not referenced for RANGE "I".
    &vu, // VU: Upper bound for singular value to search. Not referenced for RANGE "I".
    &il, // IL: Lower index of singular values returned. 1 <= IL <= IU <= min(M, N).
    &iu, // IU: Upper index of singular values returned. 1 <= IL <= IU <= min(M, N).
    &ns, // NS: Number of singular values found. Should be IU - IL + 1.
    s.data(), // S: Array holding the singular values of A.
    U.data_ptr, // U: Matrix holding left singular vectors of A.
    &ldu, // LDU: Leading dimension of U.
    Vt.data_ptr, // Vt: Matrix holding the right singular vectors of A.
    &ldvt, // LDVT: Leading dimension of Vt.
    work, // WORK: Additional scratch space.
    &lwork, // LWORK: Dimension of the array WORK.
    iwork, // IWORK: Iwork array containing indices of uncoverged elements on failure.
    &info // INFO: Exit code.
  );

  if (verbose) {
    printf("Exit code for DGESVDX: %d\n", info);
    #ifdef _MEMPRINT
    printf("Memory location of work: %p\n", work);
    printf("Memory location of iwork: %p\n", iwork);
    #endif
    A.print();
    U.print();
    Vt.print();
  }

  // Resize the singular values to k
  s.resize(k);

  // Free workspace and the copied matrix
  free(work);
  free(iwork);
  A.clear();

  return std::make_tuple(std::move(U), std::move(s), std::move(Vt));
}

Tensor slicewise_matmul(const Tensor& U, const Matrix& S, const Tensor& VT){
    // https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-0/cblas-dgmm-batch.html
    // https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-0/cblas-dgmm-batch-strided.html
    assert(U.nslices == VT.nslices);
    assert(U.nslices == S.ncol);
    assert(U.dims[1] == S.nrow);

    // Scale U with S
    std::vector<size_t> USdims(U.dims); // US is the scaled version of U
                                        // where columns of the frontal slices of U is scaled by singular values of the corresponding slice in S
                                        // Dimensions of US would be same as U
    size_t USbuflen = std::accumulate(USdims.begin(), USdims.end(), (size_t)1, std::multiplies<size_t>());
    //size_t USbuflen = 1;
    //for (int i = 0; i < USdims.size(); i++){
        //USbuflen = USbuflen * USdims[i];
    //}
    Tensor US(USbuflen, U.ndim, USdims);

    {
        // Multiplying each slice of U with the diagonal matrix corresponding to the corresponding column of matrix S (which is compact format of tensor S)
        // Use MKL ddgmm_batch_strided: https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-0/cblas-dgmm-batch-strided.html
        CBLAS_LAYOUT cblas_layout = CblasColMajor;
        CBLAS_SIDE cblas_left_right = CblasRight;
        MKL_INT cblas_m = (MKL_INT) U.dims[0];
        MKL_INT cblas_n = (MKL_INT) U.dims[1];
        double* cblas_a = U.data_ptr;
        MKL_INT cblas_lda = (MKL_INT) U.dims[0];
        MKL_INT cblas_stridea = (MKL_INT) (U.dims[0] * U.dims[1]);
        double* cblas_x = S.data_ptr;
        MKL_INT cblas_incx = 1;
        MKL_INT cblas_stridex = (MKL_INT) S.nrow; //Same as U.dims[1]
        double* cblas_c = US.data_ptr;
        MKL_INT cblas_ldc = (MKL_INT) US.dims[0];
        MKL_INT cblas_stridec = (MKL_INT) (US.dims[0] * US.dims[1]);
        MKL_INT cblas_batch_size = (MKL_INT) U.nslices;
        
        cblas_ddgmm_batch_strided(
                cblas_layout,
                cblas_left_right,
                cblas_m,
                cblas_n,
                cblas_a,
                cblas_lda,
                cblas_stridea,
                cblas_x,
                cblas_incx,
                cblas_stridex,
                cblas_c,
                cblas_ldc,
                cblas_stridec,
                cblas_batch_size
                );

    }

    std::vector<size_t> TOdims(US.dims);
    TOdims[0] = US.dims[0];
    TOdims[1] = VT.dims[1];
    //printf("TOdims[1]: %lld\n", TOdims[1]);
    size_t TObuflen = std::accumulate(TOdims.begin(), TOdims.end(), (size_t)1, std::multiplies<size_t>());
    //printf("TO buflen: %lld\n", TObuflen);
    Tensor TO(TObuflen, US.ndim, TOdims);

    {
        // https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/cblas-gemm-batch-strided.html
        CBLAS_LAYOUT cblas_layout = CblasColMajor;
        CBLAS_TRANSPOSE cblas_transa = CblasNoTrans;
        CBLAS_TRANSPOSE cblas_transb = CblasNoTrans;
        MKL_INT cblas_m = (MKL_INT) US.dims[0];
        MKL_INT cblas_k = (MKL_INT) US.dims[1]; 
        MKL_INT cblas_n = (MKL_INT) VT.dims[1];
        double cblas_alpha = 1.0;
        double cblas_beta = 0.0;
        double* cblas_a = US.data_ptr;
        MKL_INT cblas_lda = (MKL_INT) US.dims[0];
        MKL_INT cblas_stridea = (MKL_INT)(US.dims[0] * US.dims[1]);
        double* cblas_b = VT.data_ptr;
        MKL_INT cblas_ldb = (MKL_INT) VT.dims[0];
        MKL_INT cblas_strideb = (MKL_INT)(VT.dims[0] * VT.dims[1]);
        double* cblas_c = TO.data_ptr; 
        MKL_INT cblas_ldc = (MKL_INT) TO.dims[0];
        MKL_INT cblas_stridec = (MKL_INT)(TO.dims[0] * TO.dims[1]);
        MKL_INT cblas_batch_size = (MKL_INT)  TO.nslices;

        cblas_dgemm_batch_strided(
            cblas_layout, // Column major order. `Layout` parameter of MKL cblas call.
            cblas_transa, // A matrix is not transpose. `transa` param of MKL cblas call.
            cblas_transb, // B matrix is transpose. `transb` param of MKL cblas call.
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
    US.clear();
    return TO;
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
        MKL_INT cblas_m = (MKL_INT) mat_dims[0];
        MKL_INT cblas_k = (MKL_INT) ten_dims[mode]; // Or mat_dims[1]
        MKL_INT cblas_n = 1;
        for (size_t i = mode+1; i<out_ten_dims.size(); i++){
            cblas_n = cblas_n * (MKL_INT) out_ten_dims[i];
        }
        double cblas_alpha = 1.0;
        double cblas_beta = 0.0;
        double* cblas_a = M.data_ptr;
        MKL_INT cblas_lda = (MKL_INT) mat_dims[0];
        double* cblas_b = T.data_ptr;
        MKL_INT cblas_ldb = (MKL_INT) ten_dims[0];
        double* cblas_c = TO.data_ptr; 
        MKL_INT cblas_ldc = (MKL_INT) out_ten_dims[0]; // Number of rows of the matrix
                                         
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
            MKL_INT cblas_m = (MKL_INT) Mk;
            MKL_INT cblas_k = (MKL_INT) ten_dims[mode]; // Or mat_dims[1]
            MKL_INT cblas_n = (MKL_INT) mat_dims[0];
            double cblas_alpha = 1.0;
            double cblas_beta = 0.0;
            double* cblas_a = T.data_ptr + T_stride_len * l;
            MKL_INT cblas_lda = (MKL_INT) Mk;
            double* cblas_b = M.data_ptr;
            MKL_INT cblas_ldb = (MKL_INT) mat_dims[0];
            double* cblas_c = TO.data_ptr + TO_stride_len * l; 
            MKL_INT cblas_ldc = (MKL_INT) Mk;
          
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

    #ifdef _MEMPRINT
    std::cout << "ttm" << std::endl;
    std::cout << "T Pointing to: " << T.data_ptr << std::endl;
    std::cout << "M Pointing to: " << M.data_ptr << std::endl;
    std::cout << "TO Pointing to: " << TO.data_ptr << std::endl;
    #endif

    if(mode == 0) {
        // TTM on first mode
        MKL_INT cblas_m = (MKL_INT) mat_dims[0];
        MKL_INT cblas_k = (MKL_INT) ten_dims[mode]; // Or mat_dims[1]
        MKL_INT cblas_n = 1;
        for (size_t i = mode+1; i<out_ten_dims.size(); i++){
            cblas_n = cblas_n * (MKL_INT) out_ten_dims[i];
        }
        double cblas_alpha = 1.0;
        double cblas_beta = 0.0;
        double* cblas_a = M.data_ptr;
        MKL_INT cblas_lda = (MKL_INT) mat_dims[0];
        double* cblas_b = T.data_ptr;
        MKL_INT cblas_ldb = (MKL_INT) ten_dims[0];
        double* cblas_c = TO.data_ptr; 
        MKL_INT cblas_ldc = (MKL_INT) out_ten_dims[0]; // Number of rows of the matrix
                                         
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
        MKL_INT cblas_m = (MKL_INT) Mk;
        MKL_INT cblas_k = (MKL_INT) ten_dims[mode]; // Or mat_dims[1]
        MKL_INT cblas_n = (MKL_INT) mat_dims[0];
        double cblas_alpha = 1.0;
        double cblas_beta = 0.0;
        double* cblas_a = T.data_ptr;
        MKL_INT cblas_lda = (MKL_INT) Mk;
        MKL_INT cblas_stridea = (MKL_INT) (Mk * ten_dims[mode]);
        double* cblas_b = M.data_ptr;
        MKL_INT cblas_ldb = (MKL_INT) mat_dims[0];
        MKL_INT cblas_strideb = 0;
        double* cblas_c = TO.data_ptr; 
        MKL_INT cblas_ldc = (MKL_INT) Mk;
        MKL_INT cblas_stridec = (MKL_INT) (Mk * out_ten_dims[mode]);
        MKL_INT cblas_batch_size = (MKL_INT) Pk;

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

Tensor ttm_to_existing_buffer(const Tensor& T, Matrix& M, size_t mode, Tensor& TO){
    std::vector<size_t> ten_dims = T.dims;
    //auto ten_dims = T.getdims();
    std::vector<size_t> mat_dims = M.getdims();
    assert(mat_dims[1] == ten_dims[mode]);

    std::vector<size_t> out_ten_dims = TO.dims;
    //auto out_ten_dims = TO.getdims();
    //out_ten_dims[mode] = mat_dims[0];
    //size_t buflen = 1;
    //for (size_t i = 0; i<out_ten_dims.size(); i++) buflen = buflen * out_ten_dims[i];

    //Tensor TO(buflen, out_ten_dims.size(), out_ten_dims);

    #ifdef _MEMPRINT
    std::cout << "ttm" << std::endl;
    std::cout << "T Pointing to: " << T.data_ptr << std::endl;
    std::cout << "M Pointing to: " << M.data_ptr << std::endl;
    std::cout << "TO Pointing to: " << TO.data_ptr << std::endl;
    #endif

    if(mode == 0) {
        // TTM on first mode
        MKL_INT cblas_m = (MKL_INT) mat_dims[0];
        MKL_INT cblas_k = (MKL_INT) ten_dims[mode]; // Or mat_dims[1]
        MKL_INT cblas_n = 1;
        for (size_t i = mode+1; i<out_ten_dims.size(); i++){
            cblas_n = cblas_n * (MKL_INT) out_ten_dims[i];
        }
        double cblas_alpha = 1.0;
        double cblas_beta = 0.0;
        double* cblas_a = M.data_ptr;
        MKL_INT cblas_lda = (MKL_INT) mat_dims[0];
        double* cblas_b = T.data_ptr;
        MKL_INT cblas_ldb = (MKL_INT) ten_dims[0];
        double* cblas_c = TO.data_ptr; 
        MKL_INT cblas_ldc = (MKL_INT) out_ten_dims[0]; // Number of rows of the matrix
                                         
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
        MKL_INT cblas_m = (MKL_INT) Mk;
        MKL_INT cblas_k = (MKL_INT) ten_dims[mode]; // Or mat_dims[1]
        MKL_INT cblas_n = (MKL_INT) mat_dims[0];
        double cblas_alpha = 1.0;
        double cblas_beta = 0.0;
        double* cblas_a = T.data_ptr;
        MKL_INT cblas_lda = (MKL_INT) Mk;
        MKL_INT cblas_stridea = (MKL_INT) (Mk * ten_dims[mode]);
        double* cblas_b = M.data_ptr;
        MKL_INT cblas_ldb = (MKL_INT) mat_dims[0];
        MKL_INT cblas_strideb = 0;
        double* cblas_c = TO.data_ptr; 
        MKL_INT cblas_ldc = Mk;
        MKL_INT cblas_stridec = Mk * out_ten_dims[mode];
        MKL_INT cblas_batch_size = Pk;

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
    printf("Norm difference after TTM: %.16e\n", (T.norm()-TO.norm()) / T.norm() );   
    return TO;
}


Tensor transform(const Tensor& T, std::vector<Matrix> M, std::vector<int> order){
    //printf("[transform]\n");
    assert( M.size() == order.size() );
    for(int i =0; i < order.size(); i++){
        assert( order[i] < T.ndim );
    }

    size_t buflen  = 1;
    for (size_t i = 0; i < T.ndim; i++) {
      buflen = buflen * T.dims[i];
    }
    
    Tensor TO(buflen, T.ndim, T.dims);
    if(M.size() == 1){
        ttm_to_existing_buffer(T, M[0], order[0], TO);
    }
    else{
        Tensor TO_temp(buflen, T.ndim, T.dims);
        //TO_temp = T; // Should be copy
        
        //printf("[transform ]Norm diff: %0.16e\n", (T.norm()-TO_temp.norm())/T.norm());
        ttm_to_existing_buffer(T, M[0], order[0], TO);
        //if (order.size() > 1)  std::swap(TO.data_ptr, TO_temp.data_ptr);

        for(int i=1; i < order.size(); i++){
            std::swap(TO.data_ptr, TO_temp.data_ptr);
            ttm_to_existing_buffer(TO_temp, M[i], order[i], TO);
            //TO.print();
            //if(i < order.size()-1) std::swap(TO.data_ptr, TO_temp.data_ptr);
        }
        TO_temp.clear();
    }
    return TO;
}



std::tuple<Tensor, Matrix, Tensor> slicewise_svd(const Tensor &A, bool verbose=false) {
  size_t r = std::min(A.dims[0], A.dims[1]);

  // Create the variables
  std::vector<size_t> Udims = A.dims;
  Udims[1] = r;
  size_t Ubuflen = std::accumulate(Udims.begin(), Udims.end(), (size_t)1, std::multiplies<size_t>());
  Tensor U(Ubuflen, A.ndim, Udims);

  std::vector<size_t> Vtdims = A.dims;
  Vtdims[0] = r;
  size_t Vtbuflen = std::accumulate(Vtdims.begin(), Vtdims.end(), (size_t)1, std::multiplies<size_t>());
  Tensor Vt(Vtbuflen, A.ndim, Vtdims);
  Matrix S(r*A.nslices, r, A.nslices);

  // Call slice-wise SVDs
#pragma omp parallel
  {
      // Temporary slicewise SVD objects 
      //Matrix Us(Udims[0] * r, Udims[0], r);
      //Matrix Vst(r * Vtdims[1], r, Vtdims[1]);
      Matrix Us;
      Matrix Vst;
      std::vector<double> s(r);
#pragma omp for
      for (size_t i = 0; i < A.nslices; i++) {

        // Compute the SVD
        std::tie(Us, s, Vst) = svd(A.getfrontalslice_copy(i), verbose);

        // Set the output tensors
        U.setfrontalslice(Us, i);
        S.setcol(s, i);
        Vt.setfrontalslice(Vst, i);
      }

      // Clear temporary stuff
      Us.clear();
      Vst.clear();
  }


  return std::make_tuple(std::move(U), std::move(S), std::move(Vt));
}

std::tuple<Tensor, Matrix, Tensor> slicewise_svdx(const Tensor &A, size_t k,
                                                    bool verbose=false) {
  // Create the variables
  std::vector<size_t> Udims = A.dims;
  Udims[1] = k;
  size_t Ubuflen = std::accumulate(Udims.begin(), Udims.end(), (size_t)1, std::multiplies<size_t>());
  Tensor U(Ubuflen, A.ndim, Udims);

  std::vector<size_t> Vtdims = A.dims;
  Vtdims[0] = k;
  size_t Vtbuflen = std::accumulate(Vtdims.begin(), Vtdims.end(), (size_t)1, std::multiplies<size_t>());
  Tensor Vt(Vtbuflen, A.ndim, Vtdims);

  Matrix S(k*A.nslices, k, A.nslices);

  // Call slice-wise SVDs
#pragma omp parallel
  {
    // Temporary slicewise SVD objects 
    //Matrix Us(Udims[0] * k, Udims[0], k);
    //Matrix Vst(k * Vtdims[1], k, Vtdims[1]);
    Matrix Us;
    Matrix Vst;
    std::vector<double> s(k);
#pragma omp for
    for (size_t i = 0; i < A.nslices; i++) {

        // Compute the SVD
        std::tie(Us, s, Vst) = svdx(A.getfrontalslice_copy(i), k, verbose);

        // Set the output tensors
        U.setfrontalslice(Us, i);
        S.setcol(s, i);
        Vt.setfrontalslice(Vst, i);
    }

    // Clear temporary stuff
    Us.clear();
    Vst.clear();
  }

  return std::make_tuple(std::move(U), std::move(S), std::move(Vt));
}

std::tuple<Tensor, Matrix, Tensor> tsvdmi_compress(const Tensor &A, std::vector<Matrix> M, int k) {
    std::vector<int> order;
    for(int i = 0; i < A.ndim; i++){
        if(i < 2) continue;
        else order.push_back(i);
    }
    Tensor A_hat = transform(A, M, order);
    return slicewise_svdx(A_hat, k);
}

Tensor tsvdmi_reconstruct(const Tensor& U, const Matrix& S, const Tensor& VT, std::vector<Matrix> M){
    Tensor A_hat = slicewise_matmul(U, S, VT);   
    std::vector<int> order;
    for(int i = 0; i < A_hat.ndim; i++){
        if(i < 2) continue;
        else order.push_back(i);
    }
    Tensor A_tilde = transform(A_hat, M, order);
    return A_tilde;
}

void check(){
    std::vector<size_t> dims(5);
    //dims[0] = 253;
    //dims[1] = 253;
    //dims[2] = 253;
    dims[0] = 80;
    dims[1] = 80;
    dims[2] = 80;
    dims[3] = 4;
    dims[4] = 6;
    //size_t buflen = 253 * 253 * 253 * 4 * 6;
    size_t buflen = 80 * 80 * 80 * 4 * 6;
    printf("buflen: %lld\n", buflen);
    Tensor T(buflen, dims.size(), dims);
    printf("Tensor allocated\n");
    T.generate_random();
    printf("Tensor generated\n");

    double a, b, c, d;
    //a = 0.5000000000000001;
    //b = 0.6532814824381883;
    //c = 0.2705980800730985;
    a = 0.500000000000000;
    b = 0.653281482438188;
    //c = 0.270598080073099;
    c = 0.270598050073099;
    d = 1 / std::sqrt(2);
    Matrix M(16, 4, 4);
    M.set(0,0,a);
    M.set(0,1,a);
    M.set(0,2,a);
    M.set(0,3,a);
    M.set(1,0,b);
    M.set(1,1,c);
    M.set(1,2,-c);
    M.set(1,3,-b);
    M.set(2,0,a);
    M.set(2,1,-a);
    M.set(2,2,-a);
    M.set(2,3,a); 
    M.set(3,0,c); 
    M.set(3,1,-b);
    M.set(3,2,b); 
    M.set(3,3,-c);

    M.print();

    //M.set(0,0,1);
    //M.set(0,1,2e-15);
    //M.set(0,2,0);
    //M.set(0,3,0);
    //M.set(1,0,0);
    //M.set(1,1,1);
    //M.set(1,2,0);
    //M.set(1,3,0);
    //M.set(2,0,0);
    //M.set(2,1,0);
    //M.set(2,2,1);
    //M.set(2,3,0); 
    //M.set(3,0,0); 
    //M.set(3,1,0);
    //M.set(3,2,0); 
    //M.set(3,3,1);

    //M.set(0,0,d);
    //M.set(0,1,-d);
    //M.set(0,2,0);
    //M.set(0,3,0);
    //M.set(1,0,d);
    //M.set(1,1,d);
    //M.set(1,2,0);
    //M.set(1,3,0);
    //M.set(2,0,0);
    //M.set(2,1,0);
    //M.set(2,2,d);
    //M.set(2,3,-d); 
    //M.set(3,0,0); 
    //M.set(3,1,0);
    //M.set(3,2,d); 
    //M.set(3,3,d);
    
    printf("Matrix norm: %0.16e\n", M.norm());
    printf("Norm before: %0.16e\n", T.norm());
    //Tensor TO = ttm_loop(T, M, 3);
    Tensor TO = ttm(T, M, 3);
    printf("Norm after: %0.16e\n", TO.norm());
    printf("Norm diff: %0.16e\n", (TO.norm()-T.norm())/T.norm() );
    return;
}

#endif
