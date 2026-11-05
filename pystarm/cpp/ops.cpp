// ops.cpp
// Tensor and matrix operation implementation

#ifndef OPS_CPP
#define OPS_CPP

#include <numeric>
#include <tuple>
#include <stdexcept> // For std::runtime_error raised by the expm kernels
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
    //A.print();
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
    //A.print();
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
    //A.print();
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
    "N", // JOBU: Option for computing all or part of U. 'N' is no columns. 
    "N", // JOBVT: Option for computing all or part of Vt. 'N' is no rows.
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
    //A.print();
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
  size_t r   = std::min(A.nrow, A.ncol);
  size_t pad = 1; // Padding for times DBDSVDX fails
  std::vector<double> s(r+pad);

  // sizes in int for DGESVDX
  lapack_int m = A.nrow, n = A.ncol;
  lapack_int lda = m, ldu = m, ldvt = k;
  
  if (verbose) {
    //A.print();
    printf("A.nrow, A.ncol, k: %zu %zu %zu\n", A.nrow, A.ncol, k);
    printf("m, n: %d %d\n", m, n);
    printf("lda, ldu, ldvt: %d %d %d\n", lda, ldu, ldvt);
  }

  // Workspace options
  lapack_int info, lwork;
  lapack_int *iwork;
  double *work, wkopt;

  lwork = -1; // Optimal workspace query
  iwork = (lapack_int*) malloc((12 * r) * sizeof(lapack_int));

  // Options for DGESVDX
  double vl = 0.0, vu = 0.0;
  lapack_int il = 1, iu = k, ns;
  
  // Workspace query
  dgesvdx(
    "V", // JOBU: Option for computing all or part of U. 'V' is the columns specified by RANGE. 
    "V", // JOBVT: Option for computing all or part of Vt. 'V' is the rows specified by RANGE.
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
    "V", // JOBVT: Option for computing all or part of Vt. 'V' is the rows specified by RANGE.
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
    //A.print();
    //U.print();
    //Vt.print();
  }

  // Resize the singular values to k
  s.resize(k);

  // Free workspace and the copied matrix
  free(work);
  free(iwork);
  A.clear();

  return std::make_tuple(std::move(U), std::move(s), std::move(Vt));
}

double threshold(const Matrix& A, double tol, bool verbose=false) {
  // (sum of squares of discarded singular values) 
  //                  < (tol)^2 * (sum of squares of all singular values)
  assert(tol > 0.0);
  assert(tol < 1.0);
  std::vector<double> svals;
  // Working with the square of singular values
  double energy = std::pow(tol, 2); 

  // Copy values into a vector
  for (size_t i = 0; i < A.nrow; i++) {
    for (size_t j = 0; j < A.ncol; j++) {
      svals.push_back(A.get(i, j));
    }
  }
 
  // Sort and square the singular values
  std::sort(svals.begin(), svals.end());
  std::vector<double> sqsvals;
  sqsvals.resize(svals.size());
  for (size_t i = 0; i < svals.size(); i++) {
    sqsvals[i] = svals[i] * svals[i];
  }

  // Compute the partial sums
  std::partial_sum(sqsvals.begin(), sqsvals.end(), sqsvals.begin());
  for (size_t i = 0; i < sqsvals.size(); i++) {
    sqsvals[i] = sqsvals[i] / sqsvals.back();
  }

  // Find the position
  // First iterator iter in [first, last) where bool(value < *iter) 
  auto ub = std::upper_bound(sqsvals.begin(), sqsvals.end(), energy, 
              std::less<double>());

  double thr, thr_energy, thr_relerr;
  size_t s;
  if (ub != sqsvals.begin()) {
    s   = (ub - sqsvals.begin()) - 1;
    thr = svals[s];
  } else {
    thr = 0.0; // Need all the singular values!
    s   = -1;
  }

  if (verbose) {
    double thr_energy, thr_relerr;
    thr_relerr = std::sqrt(sqsvals[s]);
    thr_energy = 1.0 - sqsvals[s];
    std::cout << "Threshold        : " << thr << std::endl;
    std::cout << "Threshold index  : " << s << std::endl;
    std::cout << "Threshold energy : " << thr_energy << std::endl;
    std::cout << "Threshold relerr : " << thr_relerr << std::endl;
  }
  return thr;
}

std::vector<size_t> thresholds(const Matrix& A, double tol, bool verbose=false) {
  std::vector<size_t> slice_ranks;
  
  // Compute the minimum singular value to keep
  double thr = threshold(A, tol, verbose);

  // Find the slicewise ranks
  slice_ranks.resize(A.ncol);
  for (size_t j = 0; j < slice_ranks.size(); j++) {
    std::vector<double> col_svals;
    col_svals = A.getcol(j);

    // Find the position
    // First iterator iter in [first, last) where bool(value > *iter) 
    auto ub   = std::upper_bound(col_svals.begin(), col_svals.end(), thr, 
                  std::greater<double>());
    slice_ranks[j] = ub - col_svals.begin();
  }

  if (verbose) {
    std::cout << "Slice ranks: " << std::endl;
    for (size_t i = 0; i < slice_ranks.size(); i++)
      std::cout << slice_ranks[i] << " ";

    std::cout << std::endl;
  }
  return slice_ranks;
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

Tensor slicewise_matmul(const Tensor& A, const Tensor& B, bool transpose_A=false, bool transpose_B=false){
    // OVERLOADED OPERATOR!!! This one simply does slicewise matrix multiplication on two tensors.
    // https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-0/cblas-dgmm-batch.html
    // https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-0/cblas-dgmm-batch-strided.html
    assert(A.nslices == B.nslices);

    const size_t A_rows = transpose_A ? A.dims[1] : A.dims[0];  // m
    const size_t A_cols = transpose_A ? A.dims[0] : A.dims[1];  // k
    const size_t B_rows = transpose_B ? B.dims[1] : B.dims[0];  // k
    const size_t B_cols = transpose_B ? B.dims[0] : B.dims[1];  // n

    assert(A_cols == B_rows && "Inner dimensions must match for op(A) * op(B).");
    // Get resulting tensor dimensions.
    std::vector<size_t> TOdims(A.dims);
    // Set output dimensions based on the transpose flags supplied by the user.
    TOdims[0] = A_rows;
    TOdims[1] = B_cols;

    // printf("TOdims[1]: %lld\n", TOdims[1]);
    size_t TObuflen = std::accumulate(TOdims.begin(), TOdims.end(), (size_t)1, std::multiplies<size_t>());
    // printf("TO buflen: %lld\n", TObuflen);
    Tensor TO(TObuflen, A.ndim, TOdims);

    {
        // https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/cblas-gemm-batch-strided.html
        CBLAS_LAYOUT cblas_layout = CblasColMajor;
        CBLAS_TRANSPOSE cblas_transa = (transpose_A) ? CblasTrans: CblasNoTrans;
        CBLAS_TRANSPOSE cblas_transb = (transpose_B) ? CblasTrans: CblasNoTrans;
        MKL_INT cblas_m = (MKL_INT) A_rows;
        MKL_INT cblas_k = (MKL_INT) A_cols; 
        MKL_INT cblas_n = (MKL_INT) B_cols;
        double cblas_alpha = 1.0;
        double cblas_beta = 0.0;
        double* cblas_a = A.data_ptr;
        MKL_INT cblas_lda = (MKL_INT) A.dims[0];
        MKL_INT cblas_stridea = (MKL_INT)(A.dims[0] * A.dims[1]);
        double* cblas_b = B.data_ptr;
        MKL_INT cblas_ldb = (MKL_INT) B.dims[0];
        MKL_INT cblas_strideb = (MKL_INT)(B.dims[0] * B.dims[1]);
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
    return TO;
}

Tensor slicewise_matmul(const Tensor& U, const Matrix& S, bool inv_flag=false, bool ten_left=true){
    /* 
    Another overloaded operator, this one just does slicewise matrix multiply on a tensor and a diagonal tensor stored as a matrix.
    // https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-0/cblas-dgmm-batch.html
    // https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-0/cblas-dgmm-batch-strided.html

    PARAMS:
    U: Tensor of size (m x n x n_slices)
    S: Diagonal tensor stored as matrix of size (n x n_slices)
    inv_flag: Whether or not you want matrix S to be inverse. 
    ten_left: True if doing U * S, False if doing S * U
    */
    assert(U.nslices == S.ncol);
    assert(U.dims[1] == S.nrow);
    // assert(!(inv_flag == true && ten_left == true));
    if(!ten_left){
      assert(U.dims[0] == S.nrow);
    }

    // Scale U with S
    std::vector<size_t> USdims(U.dims); // US is the scaled version of U
                                        // where columns of the frontal slices of U is scaled by singular values of the corresponding slice in S
                                        // Dimensions of US would be same as U
    USdims[1] = S.nrow; // Because we are not multiplying by VT in the end, we will not have the same dimensions as U.
    size_t USbuflen = std::accumulate(USdims.begin(), USdims.end(), (size_t)1, std::multiplies<size_t>());
    //size_t USbuflen = 1;
    //for (int i = 0; i < USdims.size(); i++){
        //USbuflen = USbuflen * USdims[i];
    //}
    Tensor US(USbuflen, U.ndim, USdims);
    Matrix S_inv(S.nrow * S.ncol, S.nrow, S.ncol);
    if(inv_flag){
      for (size_t i = 0; i < S_inv.buflen; ++i){
        for(size_t j = 0; j < S_inv.buflen; ++j){
          S_inv.set(i, j, 1 / S.get(i, j));
        }
      }
    }
    // Multiplying each slice of U with the diagonal matrix corresponding to the corresponding column of matrix S (which is compact format of tensor S)
    // Use MKL ddgmm_batch_strided: https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-0/cblas-dgmm-batch-strided.html
    CBLAS_LAYOUT cblas_layout = CblasColMajor;
    CBLAS_SIDE cblas_left_right = (ten_left) ? CblasRight: CblasLeft;
    MKL_INT cblas_m = (MKL_INT) U.dims[0];
    MKL_INT cblas_n = (MKL_INT) U.dims[1];
    double* cblas_a = U.data_ptr;
    MKL_INT cblas_lda = (MKL_INT) U.dims[0];
    MKL_INT cblas_stridea = (MKL_INT) (U.dims[0] * U.dims[1]);
    double* cblas_x = (inv_flag) ? S_inv.data_ptr : S.data_ptr;
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

      // #pragma omp parallel for simd schedule(static) if(USbuflen > 100000)
      // for(size_t j = 0; j < S.ncol; ++j){
      //   for(size_t i = 0; i < S.nrow; ++i){
      //     size_t U_index = ((i % U_n_cols) + (i * U_n_cols));
      //     US.data_ptr[U_index] = U.data_ptr[U_index] / S.get(i, j);
      //   }
    S_inv.clear();
    return US;
  }


Tensor slicewise_matmulks(const JaggedTensor& U, const JaggedMatrix& S, const JaggedTensor& VT){
    // https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-0/cblas-dgmm-batch.html
    // https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-0/cblas-dgmm-batch-strided.html
    assert(U.nslices == VT.nslices);
    assert(U.nslices == S.ncol);
    
    // US would have same size as U just scaled by S
    size_t USbuflen = 0;
    for (size_t i = 0; i < U.nslices; i++){
        USbuflen = USbuflen + U.fixed_dim_size * U.slice_ranks[i];
    }
    JaggedTensor US(USbuflen, U.fixed_dim_size, U.slice_ranks, U.variable_first_mode);

    {
        // Multiplying each slice of U with the diagonal matrix corresponding to the corresponding column of matrix S (which is compact format of tensor S)
        // Use MKL ddgmm_batch: https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-0/cblas-dgmm-batch.html
        MKL_INT cblas_group_count = (MKL_INT)US.nslices;
        MKL_INT* cblas_group_size = new MKL_INT[cblas_group_count];

        CBLAS_LAYOUT cblas_layout = CblasColMajor;
        CBLAS_SIDE* cblas_left_right_array = new CBLAS_SIDE[cblas_group_count];
        MKL_INT* cblas_m_array = new MKL_INT[cblas_group_count];
        MKL_INT* cblas_n_array = new MKL_INT[cblas_group_count];
        const double** cblas_a_array = new const double*[cblas_group_count];
        MKL_INT* cblas_lda_array = new MKL_INT[cblas_group_count];
        const double** cblas_x_array = new const double*[cblas_group_count];
        MKL_INT* cblas_incx_array = new MKL_INT[cblas_group_count];
        double** cblas_c_array = new double*[cblas_group_count];
        MKL_INT* cblas_ldc_array = new MKL_INT[cblas_group_count];

        for(MKL_INT i = 0; i < cblas_group_count; i++){ 
            cblas_left_right_array[i] = CblasRight; 
            cblas_m_array[i] = US.fixed_dim_size; 
            cblas_n_array[i] = US.slice_ranks[i]; 
            if (i == 0)
                cblas_a_array[i] = U.data_ptr ; 
            else
                cblas_a_array[i] = cblas_a_array[i-1] + U.fixed_dim_size * U.slice_ranks[i-1] ; 
            cblas_lda_array[i] = U.fixed_dim_size; 
            if (i == 0)
                cblas_x_array[i] = S.data_ptr ; 
            else
                cblas_x_array[i] = cblas_x_array[i-1] +  US.slice_ranks[i-1] ; 
            cblas_incx_array[i] = 1; 
            if (i == 0)
                cblas_c_array[i] = US.data_ptr; 
            else
                cblas_c_array[i] = cblas_c_array[i-1] + US.fixed_dim_size * US.slice_ranks[i-1] ; 
            cblas_ldc_array[i] = US.fixed_dim_size; 
            cblas_group_size[i] = 1;
        }

        cblas_ddgmm_batch (
                cblas_layout, 
                cblas_left_right_array, 
                cblas_m_array, 
                cblas_n_array, 
                cblas_a_array, 
                cblas_lda_array, 
                cblas_x_array, 
                cblas_incx_array, 
                cblas_c_array, 
                cblas_ldc_array, 
                cblas_group_count, 
                cblas_group_size
        );

        delete[] cblas_left_right_array;
        delete[] cblas_m_array;
        delete[] cblas_n_array;
        delete[] cblas_a_array;
        delete[] cblas_lda_array;
        delete[] cblas_x_array;
        delete[] cblas_incx_array;
        delete[] cblas_c_array;
        delete[] cblas_ldc_array;
        delete[] cblas_group_size;

    }
    
    // Because JaggedTensors are simplified as 3-way tensors, TO would be a 3-way regular tensor
    std::vector<size_t> TOdims(3);
    TOdims[0] = US.fixed_dim_size;
    TOdims[1] = VT.fixed_dim_size;
    TOdims[2] = US.nslices;
    size_t TObuflen = std::accumulate(TOdims.begin(), TOdims.end(), (size_t)1, std::multiplies<size_t>());
    Tensor TO(TObuflen, 3, TOdims);
    {
        MKL_INT cblas_m;
        MKL_INT cblas_k;
        MKL_INT cblas_n;
        double* cblas_a;
        double* cblas_b;
        double* cblas_c;
        MKL_INT cblas_lda;
        MKL_INT cblas_ldb;
        MKL_INT cblas_ldc;
        double cblas_alpha = 1.0;
        double cblas_beta = 0.0;

        for (size_t i = 0; i < US.nslices; i++){
            cblas_m = (MKL_INT) US.fixed_dim_size; 
            cblas_k = (MKL_INT) US.slice_ranks[i]; 
            cblas_n = (MKL_INT) VT.fixed_dim_size; 
            cblas_alpha = 1.0;
            cblas_beta = 0.0;
            if (i == 0)
                cblas_a = US.data_ptr; 
            else
                cblas_a = cblas_a + US.fixed_dim_size * US.slice_ranks[i-1] ; 

            cblas_lda = (MKL_INT)US.fixed_dim_size; 

            if (i == 0)
                cblas_b = VT.data_ptr; 
            else
                cblas_b = cblas_b + VT.fixed_dim_size * VT.slice_ranks[i-1] ; 

            cblas_ldb = VT.slice_ranks[i]; 

            if (i == 0)
                cblas_c = TO.data_ptr; 
            else
                cblas_c = cblas_c + US.fixed_dim_size * VT.fixed_dim_size ; 

            cblas_ldc = US.fixed_dim_size; 
                                                 
            if(US.slice_ranks[i] > 0){
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
        }
    }

    //{
        //// Use cblas_dgemm_batch: https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/cblas-gemm-batch.html
        ////
        
        //MKL_INT cblas_group_count = (MKL_INT)US.nslices;
        //MKL_INT* cblas_group_size = new MKL_INT[cblas_group_count];

        //CBLAS_LAYOUT cblas_layout = CblasColMajor;
        //CBLAS_TRANSPOSE* cblas_transa_array = new CBLAS_TRANSPOSE[cblas_group_count];
        //CBLAS_TRANSPOSE* cblas_transb_array = new CBLAS_TRANSPOSE[cblas_group_count];
        //MKL_INT* cblas_m_array = new MKL_INT[cblas_group_count];
        //MKL_INT* cblas_n_array = new MKL_INT[cblas_group_count];
        //MKL_INT* cblas_k_array = new MKL_INT[cblas_group_count];
        //double* cblas_alpha_array = new double[cblas_group_count];
        //const double** cblas_a_array = new const double*[cblas_group_count];
        //MKL_INT* cblas_lda_array = new MKL_INT[cblas_group_count];
        //const double** cblas_b_array = new const double*[cblas_group_count];
        //MKL_INT* cblas_ldb_array = new MKL_INT[cblas_group_count];
        //double* cblas_beta_array = new double[cblas_group_count];
        //double** cblas_c_array = new double*[cblas_group_count];
        //MKL_INT* cblas_ldc_array = new MKL_INT[cblas_group_count];

        //for(MKL_INT i = 0; i < cblas_group_count; i++){ 
            //cblas_group_size[i] = 1;
            //cblas_transa_array[i] = CblasNoTrans;
            //cblas_transb_array[i] = CblasNoTrans;
            //cblas_m_array[i] = US.fixed_dim_size; 
            //cblas_k_array[i] = US.slice_ranks[i]; 
            //cblas_n_array[i] = VT.fixed_dim_size; 

            //if (i == 0)
                //cblas_a_array[i] = US.data_ptr; 
            //else
                //cblas_a_array[i] = cblas_a_array[i-1] + US.fixed_dim_size * US.slice_ranks[i-1] ; 

            //cblas_lda_array[i] = US.fixed_dim_size; 

            //if (i == 0)
                //cblas_b_array[i] = VT.data_ptr; 
            //else
                //cblas_b_array[i] = cblas_b_array[i-1] + VT.fixed_dim_size * VT.slice_ranks[i-1] ; 

            //cblas_ldb_array[i] = VT.slice_ranks[i]; 

            //if (i == 0)
                //cblas_c_array[i] = TO.data_ptr; 
            //else
                //cblas_c_array[i] = cblas_c_array[i-1] + US.fixed_dim_size * VT.fixed_dim_size ; 
            //cblas_ldc_array[i] = US.fixed_dim_size; 

            //cblas_alpha_array[i] = 1.0;
            //cblas_beta_array[i] = 0.0;
        //}
        
        //cblas_dgemm_batch (
                //cblas_layout, 
                //cblas_transa_array, 
                //cblas_transb_array, 
                //cblas_m_array, 
                //cblas_n_array, 
                //cblas_k_array, 
                //cblas_alpha_array, 
                //cblas_a_array, 
                //cblas_lda_array, 
                //cblas_b_array, 
                //cblas_ldb_array, 
                //cblas_beta_array, 
                //cblas_c_array, 
                //cblas_ldc_array, 
                //cblas_group_count, 
                //cblas_group_size
        //);

        //delete[] cblas_transa_array;
        //delete[] cblas_transb_array;
        //delete[] cblas_m_array;
        //delete[] cblas_n_array;
        //delete[] cblas_k_array;
        //delete[] cblas_alpha_array;
        //delete[] cblas_a_array;
        //delete[] cblas_lda_array;
        //delete[] cblas_b_array;
        //delete[] cblas_ldb_array;
        //delete[] cblas_beta_array;
        //delete[] cblas_c_array;
        //delete[] cblas_ldc_array;
        //delete[] cblas_group_size;
    //}
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

Tensor ttm_parfor(Tensor& T, Matrix& M, size_t mode){
    std::vector<size_t> ten_dims = T.getdims();
    std::vector<size_t> mat_dims = M.getdims();
    assert(mat_dims[1] == ten_dims[mode]);

    std::vector<size_t> out_ten_dims(ten_dims);
    out_ten_dims[mode] = mat_dims[0];
    size_t buflen = 1;
    for (size_t i = 0; i<out_ten_dims.size(); i++) buflen = buflen * out_ten_dims[i];

    Tensor TO(buflen, out_ten_dims.size(), out_ten_dims);

    if(mode == 0) {
        // TTM on first mode — identical to ttm_loop
        MKL_INT cblas_m = (MKL_INT) mat_dims[0];
        MKL_INT cblas_k = (MKL_INT) ten_dims[mode];
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
        MKL_INT cblas_ldc = (MKL_INT) out_ten_dims[0];

        cblas_dgemm(
            CblasColMajor,
            CblasNoTrans,
            CblasNoTrans,
            cblas_m,
            cblas_n,
            cblas_k,
            cblas_alpha,
            cblas_a,
            cblas_lda,
            cblas_b,
            cblas_ldb,
            cblas_beta,
            cblas_c,
            cblas_ldc
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
        size_t T_stride_len = Mk * ten_dims[mode];
        size_t TO_stride_len = Mk * out_ten_dims[mode];

#pragma omp parallel
        {
            mkl_set_num_threads_local(1);
#pragma omp for
            for(size_t l = 0; l < Pk; l++){
                MKL_INT cblas_m = (MKL_INT) Mk;
                MKL_INT cblas_k = (MKL_INT) ten_dims[mode];
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
                    CblasColMajor,
                    CblasNoTrans,
                    CblasTrans,
                    cblas_m,
                    cblas_n,
                    cblas_k,
                    cblas_alpha,
                    cblas_a,
                    cblas_lda,
                    cblas_b,
                    cblas_ldb,
                    cblas_beta,
                    cblas_c,
                    cblas_ldc
                );
            }
            mkl_set_num_threads_local(0);
        }
    }

    return TO;
}
/* TODO: modify to have an m_transpose flag here. Ensure correct dimension checking and whatnot. */
Tensor ttm(Tensor& T, Matrix& M, size_t mode, bool M_transpose=false){
    std::vector<size_t> ten_dims = T.getdims();
    std::vector<size_t> mat_dims = M.getdims();
    if(M_transpose){
      assert(mat_dims[0] == ten_dims[mode]);
      // mat_dims = [mat_dims[1], mat_dims[0]];
    }
    else {
      assert(mat_dims[1] == ten_dims[mode]);
    }
    

    std::vector<size_t> out_ten_dims(ten_dims);
    out_ten_dims[mode] = (M_transpose) ? mat_dims[1] : mat_dims[0];
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
        CBLAS_TRANSPOSE transa = (M_transpose) ? CblasTrans : CblasNoTrans;
        CBLAS_TRANSPOSE transb = CblasNoTrans;
        MKL_INT cblas_m = (MKL_INT) (M_transpose) ? mat_dims[1] : mat_dims[0];
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
        MKL_INT cblas_ldc = (MKL_INT) (M_transpose) ? mat_dims[1]: mat_dims[0] ; // Number of rows of the matrix
                                         
        cblas_dgemm(
            CblasColMajor, // Column major order. `Layout` parameter of MKL cblas call.
            transa, // A matrix is not transpose. `transa` param of MKL cblas call.
            transb, // B matrix is not transpose. `transb` param of MKL cblas call.
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
        CBLAS_TRANSPOSE transa = CblasNoTrans;
        CBLAS_TRANSPOSE transb = (M_transpose) ? CblasNoTrans : CblasTrans; // Inverted because normally you multiply by M^T. See algorithm 3.1 of tensor textbook
        MKL_INT cblas_m = (MKL_INT) Mk;
        MKL_INT cblas_k = (MKL_INT) ten_dims[mode]; // Or mat_dims[1]
        MKL_INT cblas_n = (MKL_INT) (M_transpose) ? mat_dims[1] : mat_dims[0];
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
            transa, // A matrix is not transpose. `transa` param of MKL cblas call.
            transb, // B matrix is transpose. `transb` param of MKL cblas call.
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
      mkl_set_num_threads_local(1);
      // Temporary slicewise SVD objects
      //Matrix Us(Udims[0] * r, Udims[0], r);
      //Matrix Vst(r * Vtdims[1], r, Vtdims[1]);
      Matrix Us;
      Matrix Vst;
      std::vector<double> s(r);
#pragma omp for
      for (size_t i = 0; i < A.nslices; i++) {

        // Compute the SVD
        //double t0 = omp_get_wtime();
        std::tie(Us, s, Vst) = svd(A.getfrontalslice_copy(i), verbose);
        //double t1 = omp_get_wtime();
        //printf("[slicewise_svd] slice %zu: %.6f s\n", i, t1 - t0);

        // Set the output tensors
        U.setfrontalslice(Us, i);
        S.setcol(s, i);
        Vt.setfrontalslice(Vst, i);
      }

      // Clear temporary stuff
      Us.clear();
      Vst.clear();
      mkl_set_num_threads_local(0);
  }

  return std::make_tuple(std::move(U), std::move(S), std::move(Vt));
}

/* TODO: Need to work around instabilities in MKL SVD batch strided
std::tuple<Tensor, Matrix, Tensor> slicewise_svd_mkl(const Tensor &A, bool verbose=false) {
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

  // https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-fortran/2025-0/gesvda-batch-strided.html

  // Input parameters
  size_t nparams  = 16;

  MKL_INT *iparm, *irank;
  iparm = (MKL_INT*) malloc(nparams * sizeof(MKL_INT));
  irank = (MKL_INT*) malloc(A.nslices * sizeof(MKL_INT));

  MKL_INT m, n, lda, stride_a, stride_s, ldu, stride_u, ldvt, stride_vt;
  MKL_INT batch_size, lwork, info;

  double *work, wkopt, tol;

  // Set params
  iparm[0] = 0; // Compute truncated SVD with help of array irank
  iparm[1] = 0; // Only compute singular values
  iparm[2] = 0; // SVD is computed as product of three matrices (should not matter)
  iparm[3] = 0; // Residual not computed

  // Compute all singular values
  for (size_t ii = 0; ii < A.nslices; ii++) {
    irank[ii] = r;
  }
  m          = (MKL_INT) A.dims[0];
  n          = (MKL_INT) A.dims[1];
  lda        = (MKL_INT) A.dims[0];
  stride_a   = m * n;
  stride_s   = (MKL_INT) r;
  ldu        = (MKL_INT) A.dims[0];
  stride_u   = m * m;
  ldvt       = (MKL_INT) r;
  stride_vt  = n * n;
  batch_size = (MKL_INT) A.nslices;
  tol        = 0.0;

  // Workspace query
  lwork = -1;

  dgesvda_batch_strided(
    iparm,        // iparm: Options for truncated SVD
    irank,        // irank: Specifies ranks to compute for each batch
    &m,           // m: No. of rows in the matrices A_i
    &n,           // n: No. of columns in the matrices A_i
    A.data_ptr,   // A: Array of input matrices A_i
    &lda,         // lda: Leading dimension of A_i
    &stride_a,    // stride_a: Stride between two A_i matrices
    S.data_ptr,   // S: Array containing the S_i matrices
    &stride_s,    // stride_s: Stride between two S_i matrices
    U.data_ptr,   // U: Array to hold U_i matrices (not needed)
    &ldu,         // ldu: Leading dimension of U_i
    &stride_u,    // stride_u: Stride between two U_i matrices
    Vt.data_ptr,  // Vt: Array to hold Vt_i matrices (not needed)
    &ldvt,        // ldvt: Leading dimension of Vt_i
    &stride_vt,   // stride_vt: Stride between two Vt_i matrices
    &tol,         // tolerance: Used for stopping SVD (not referenced)
    nullptr,      // residual: Store residuals here (not referenced)
    &wkopt,       // work: Scratch space
    &lwork,       // lwork: Dimension of the array work
    &batch_size,  // batch_size: No. of problems in a batch
    &info         // info: Exit code.
  );

  lwork = (MKL_INT) wkopt;
  work  = (double*) malloc(lwork * sizeof(double));

  if (verbose) {
    std::cout << "Size of the work array: " << lwork << std::endl;
  }

  // Compute the batch SVD
  dgesvda_batch_strided(
    iparm,        // iparm: Options for truncated SVD
    irank,        // irank: Specifies ranks to compute for each batch
    &m,            // m: No. of rows in the matrices A_i
    &n,            // n: No. of columns in the matrices A_i
    A.data_ptr,   // A: Array of input matrices A_i
    &lda,          // lda: Leading dimension of A_i
    &stride_a,     // stride_a: Stride between two A_i matrices
    S.data_ptr,   // S: Array containing the S_i matrices
    &stride_s,     // stride_s: Stride between two S_i matrices
    U.data_ptr,   // U: Array to hold U_i matrices (not needed)
    &ldu,          // ldu: Leading dimension of U_i
    &stride_u,     // stride_u: Stride between two U_i matrices
    Vt.data_ptr,  // Vt: Array to hold Vt_i matrices (not needed)
    &ldvt,         // ldvt: Leading dimension of Vt_i
    &stride_vt,    // stride_vt: Stride between two Vt_i matrices
    &tol,          // tolerance: Used for stopping SVD (not referenced)
    nullptr,      // residual: Store residuals here (not referenced)
    work,         // work: Scratch space
    &lwork,       // lwork: Dimension of the array work
    &batch_size,   // batch_size: No. of problems in a batch
    &info          // info: Exit code.
  );

  // Free temporaries
  free(work);
  free(iparm);
  free(irank);

  return std::make_tuple(std::move(U), std::move(S), std::move(Vt));
}
*/

// slicewise_svd_seq: sequential variant of slicewise_svd for benchmarking purposes.
// Processes one slice at a time (no OMP parallelism), so each dgesvd call inside
// svd() runs with all available MKL threads. Contrast with slicewise_svd which
// runs OMP threads in parallel each doing a single-threaded dgesvd.
std::tuple<Tensor, Matrix, Tensor> slicewise_svd_seq(const Tensor &A, bool verbose=false) {
  size_t r = std::min(A.dims[0], A.dims[1]);

  std::vector<size_t> Udims = A.dims;
  Udims[1] = r;
  size_t Ubuflen = std::accumulate(Udims.begin(), Udims.end(), (size_t)1, std::multiplies<size_t>());
  Tensor U(Ubuflen, A.ndim, Udims);

  std::vector<size_t> Vtdims = A.dims;
  Vtdims[0] = r;
  size_t Vtbuflen = std::accumulate(Vtdims.begin(), Vtdims.end(), (size_t)1, std::multiplies<size_t>());
  Tensor Vt(Vtbuflen, A.ndim, Vtdims);
  Matrix S(r*A.nslices, r, A.nslices);

  Matrix Us;
  Matrix Vst;
  std::vector<double> s(r);

  //printf("[slicewise_svd_seq] mkl_get_max_threads = %d, mkl_domain_get_max_threads(LAPACK) = %d\n",
  //       mkl_get_max_threads(), mkl_domain_get_max_threads(MKL_DOMAIN_LAPACK));

  //double t0, t1;
  for (size_t i = 0; i < A.nslices; i++) {
    //t0 = omp_get_wtime();
    std::tie(Us, s, Vst) = svd(A.getfrontalslice_copy(i), verbose);
    //t1 = omp_get_wtime();
    //printf("[slicewise_svd_seq] slice %zu: %.6f s\n", i, t1 - t0);
    U.setfrontalslice(Us, i);
    S.setcol(s, i);
    Vt.setfrontalslice(Vst, i);
  }

  Us.clear();
  Vst.clear();

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
    mkl_set_num_threads_local(1);
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
    mkl_set_num_threads_local(0);
  }

  return std::make_tuple(std::move(U), std::move(S), std::move(Vt));
}

Matrix slicewise_svdvals(const Tensor &A, bool verbose=false) {
  size_t r = std::min(A.dims[0], A.dims[1]);
  Matrix S(r*A.nslices, r, A.nslices);

  // Call slice-wise SVD values
#pragma omp parallel
  {
      mkl_set_num_threads_local(1);
      // Temporary slicewise SVD objects
      std::vector<double> s(r);
#pragma omp for
      for (size_t i = 0; i < A.nslices; i++) {

        // Compute the SVD
        s = svdvals(A.getfrontalslice_copy(i), verbose);

        // Set the output tensors
        S.setcol(s, i);
      }
      mkl_set_num_threads_local(0);
  }

  return S;
}

/* TODO: Need to work around instabilities in MKL SVD batch strided
Matrix slicewise_svdvals_mkl(const Tensor &A, bool verbose=false) {
  size_t r = std::min(A.dims[0], A.dims[1]);
  Matrix S(r*A.nslices, r, A.nslices);

  // https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-fortran/2025-0/gesvda-batch-strided.html

  // Input parameters
  size_t nparams = 16;

  MKL_INT *iparm, *irank;
  iparm = (MKL_INT*) malloc(nparams * sizeof(MKL_INT));
  irank = (MKL_INT*) malloc(A.nslices * sizeof(MKL_INT));

  MKL_INT m, n, lda, stride_a, stride_s, ldu, stride_u, ldvt, stride_vt;
  MKL_INT batch_size, lwork, info;

  double *work, wkopt, tol;

  // Set params
  iparm[0] = 0; // Compute truncated SVD with help of array irank
  iparm[1] = 1; // Only compute singular values
  iparm[2] = 0; // SVD is computed as product of three matrices (should not matter)
  iparm[3] = 0; // Residual not computed

  // Compute all singular values
  for (size_t ii = 0; ii < A.nslices; ii++) {
    irank[ii] = r;
  }
  m          = (MKL_INT) A.dims[0];
  n          = (MKL_INT) A.dims[1];
  lda        = (MKL_INT) A.dims[0];
  stride_a   = m * n;
  stride_s   = (MKL_INT) r;
  ldu        = (MKL_INT) A.dims[0];
  stride_u   = m * m;
  ldvt       = (MKL_INT) r;
  stride_vt  = n * n;
  batch_size = (MKL_INT) A.nslices;
  tol        = 0.0;

  // Workspace query
  lwork = -1;

  dgesvda_batch_strided(
    iparm,        // iparm: Options for truncated SVD
    irank,        // irank: Specifies ranks to compute for each batch
    &m,           // m: No. of rows in the matrices A_i
    &n,           // n: No. of columns in the matrices A_i
    A.data_ptr,   // A: Array of input matrices A_i
    &lda,         // lda: Leading dimension of A_i
    &stride_a,    // stride_a: Stride between two A_i matrices
    S.data_ptr,   // S: Array containing the S_i matrices
    &stride_s,    // stride_s: Stride between two S_i matrices
    nullptr,      // U: Array to hold U_i matrices (not needed)
    &ldu,         // ldu: Leading dimension of U_i
    &stride_u,    // stride_u: Stride between two U_i matrices
    nullptr,      // Vt: Array to hold Vt_i matrices (not needed)
    &ldvt,        // ldvt: Leading dimension of Vt_i
    &stride_vt,   // stride_vt: Stride between two Vt_i matrices
    &tol,         // tolerance: Used for stopping SVD (not referenced)
    nullptr,      // residual: Store residuals here (not referenced)
    &wkopt,       // work: Scratch space
    &lwork,       // lwork: Dimension of the array work
    &batch_size,  // batch_size: No. of problems in a batch
    &info         // info: Exit code.
  );

  lwork = (MKL_INT) wkopt;
  work  = (double*) malloc(lwork * sizeof(double));

  if (verbose) {
    std::cout << "Size of the work array: " << lwork << std::endl;
  }

  // Compute the batch SVD
  dgesvda_batch_strided(
    iparm,        // iparm: Options for truncated SVD
    irank,        // irank: Specifies ranks to compute for each batch
    &m,           // m: No. of rows in the matrices A_i
    &n,           // n: No. of columns in the matrices A_i
    A.data_ptr,   // A: Array of input matrices A_i
    &lda,         // lda: Leading dimension of A_i
    &stride_a,    // stride_a: Stride between two A_i matrices
    S.data_ptr,   // S: Array containing the S_i matrices
    &stride_s,    // stride_s: Stride between two S_i matrices
    nullptr,      // U: Array to hold U_i matrices (not needed)
    &ldu,         // ldu: Leading dimension of U_i
    &stride_u,    // stride_u: Stride between two U_i matrices
    nullptr,      // Vt: Array to hold Vt_i matrices (not needed)
    &ldvt,        // ldvt: Leading dimension of Vt_i
    &stride_vt,   // stride_vt: Stride between two Vt_i matrices
    &tol,         // tolerance: Used for stopping SVD (not referenced)
    nullptr,      // residual: Store residuals here (not referenced)
    work,         // work: Scratch space
    &lwork,       // lwork: Dimension of the array work
    &batch_size,  // batch_size: No. of problems in a batch
    &info         // info: Exit code.
  );

  // Free temporaries
  free(work);
  free(iparm);
  free(irank);

  return S;
}
*/

std::tuple<JaggedTensor, JaggedMatrix, JaggedTensor> truncate_factors(
  const Tensor &U, const Matrix &S, const Tensor &Vt, std::vector<size_t> ks) {
  // Create the variables
  size_t Uk_fixed_dim = U.dims[0];
  size_t Uk_buflen    = 0;
  for (size_t r = 0; r < ks.size(); r++) {
    Uk_buflen += Uk_fixed_dim * ks[r];
  }
  JaggedTensor Uk(Uk_buflen, Uk_fixed_dim, ks);

  size_t Vkt_fixed_dim = Vt.dims[1];
  size_t Vkt_buflen    = 0;
  for (size_t r = 0; r < ks.size(); r++) {
    Vkt_buflen += Vkt_fixed_dim * ks[r];
  }
  JaggedTensor Vkt(Vkt_buflen, Vkt_fixed_dim, ks, true);

  size_t Sk_buflen = std::accumulate(ks.begin(), ks.end(), (size_t) 0);
  JaggedMatrix Sk(Sk_buflen, ks);

  // Truncate the factors
#pragma omp parallel
  {
    // Temporary slicewise SVD objects
    Matrix Us, Uks;
    Matrix Vst, Vkst;
#pragma omp for
    for (size_t i = 0; i < ks.size(); i++) {
      if (ks[i] > 0) {
        // Get copies to the full tensors
        // TODO: need to modify interface to get views of const Tensor
        Us  = U.getfrontalslice_copy(i);
        Vst = Vt.getfrontalslice_copy(i);

        // Note: U matrix need not be copied (can be optimised later)
        // Vt matrix needs to be copied (due to column-major ordering)
        Uks  = Matrix(Uk_fixed_dim * ks[i], Uk_fixed_dim, ks[i]);
        Vkst = Matrix(ks[i] * Vkt_fixed_dim, ks[i], Vkt_fixed_dim);

        for (size_t rr = 0; rr < ks[i]; rr++) {
          Uks.setcol(Us.getcol(rr), rr);
          Vkst.setrow(Vst.getrow(rr), rr);
        }

        std::vector<double> s = S.getcol(i);
        s.resize(ks[i]);

        // Set the output tensors
        Uk.setfrontalslice(Uks, i);
        Sk.setcol(s, i);
        Vkt.setfrontalslice(Vkst, i);
      }
    }

    // Clear temporary stuff
    Uks.clear();
    Vkst.clear();
    Us.clear();
    Vst.clear();
  }

  return std::make_tuple(std::move(Uk), std::move(Sk), std::move(Vkt));
}

std::tuple<JaggedTensor, JaggedMatrix, JaggedTensor> slicewise_svdks(
  const Tensor &A, std::vector<size_t> ks, bool verbose=false) {
  // Create the variables
  size_t U_fixed_dim = A.dims[0];
  size_t U_buflen    = 0;
  for (size_t r = 0; r < ks.size(); r++) {
    U_buflen += U_fixed_dim * ks[r];
  }
  JaggedTensor U(U_buflen, U_fixed_dim, ks);

  size_t Vt_fixed_dim = A.dims[1];
  size_t Vt_buflen    = 0;
  for (size_t r = 0; r < ks.size(); r++) {
    Vt_buflen += Vt_fixed_dim * ks[r];
  }
  JaggedTensor Vt(Vt_buflen, Vt_fixed_dim, ks, true);

  size_t S_buflen = std::accumulate(ks.begin(), ks.end(), (size_t) 0);
  JaggedMatrix S(S_buflen, ks);

  // Call slice-wise SVDs
#pragma omp parallel
  {
    mkl_set_num_threads_local(1);
    // Temporary slicewise SVD objects
    Matrix Us;
    Matrix Vst;
#pragma omp for
    for (size_t i = 0; i < A.nslices; i++) {
      if (ks[i] > 0) {
        std::vector<double> s(ks[i]);

        // Compute the SVD
        std::tie(Us, s, Vst) = svdx(A.getfrontalslice_copy(i), ks[i], verbose);

        // Set the output tensors
        U.setfrontalslice(Us, i);
        S.setcol(s, i);
        Vt.setfrontalslice(Vst, i);
      }
    }

    // Clear temporary stuff
    Us.clear();
    Vst.clear();
    mkl_set_num_threads_local(0);
  }

  return std::make_tuple(std::move(U), std::move(S), std::move(Vt));
}

std::tuple<JaggedTensor, JaggedMatrix, JaggedTensor> slicewise_svd_thr(
  const Tensor &A, double tol, bool verbose=false) {

  #pragma omp barrier
  double t0, t1;
  t0 = omp_get_wtime();

  // STAGE 0: Preprocessing
  //Tensor A_copy(A); // call a copy tensor
  Tensor A_copy(A.buflen, A.ndim, A.dims);

  /*
#pragma omp for
  for (size_t ii = 0; ii < A.nslices; ii++) {
    Matrix As = A.getfrontalslice_copy(ii);
    A_copy.setfrontalslice(A.getfrontalslice_copy(ii), ii);
    As.clear();
  }
  */

  // Scale the data
  // Get the machine constants
  double eps = dlamch("P");
  double smlnum = std::sqrt(dlamch("S")) / eps;
  double bignum = 1.0 / smlnum;

  //std::cout << eps << " " << smlnum << " " << bignum << std::endl;
  // Scale A if max element outside range [smlnum, bignum]
  lapack_int li_buflen = A_copy.buflen;
  lapack_int li_one = 1, li_zero = 0;
  lapack_int info = 0;
  // Get max(abs(a(i,j)))
  double anrm = dlange("M", &li_buflen, &li_one, 
                  A_copy.data_ptr, &li_buflen, nullptr);

  bool lscl = false;
  if (anrm > 0.0 && anrm < smlnum) {
    lscl = true;
    //std::cout << "Reached SMLNUM" << std::endl;
    dlascl(
      "G",              // G: A is a full matrix.
      &li_zero,         // KL: Not referenced.
      &li_zero,         // KU: Not referenced.
      &anrm,            // CFROM: Matrix scaled as A(i, j) * CTO/CFROM.
      &smlnum,          // CTO: Matrix scaled as A(i, j) * CTO/CFROM.
      &li_buflen,       // M: No. of rows of the matrix. Treat as a long array.
      &li_one,          // N: No. of columns of the matrix. Treat as a long array.
      A_copy.data_ptr,  // A: The data array.
      &li_buflen,       // LDA: Leading dimension of A.
      &info             // INFO: Exit code.
    );
  } else if (anrm > bignum) {
    lscl = true;
    //std::cout << "Reached BIGNUM" << std::endl;
    dlascl( 
      "G",              // G: A is a full matrix.
      &li_zero,         // KL: Not referenced.
      &li_zero,         // KU: Not referenced.
      &anrm,            // CFROM: Matrix scaled as A(i, j) * CTO/CFROM.
      &bignum,          // CTO: Matrix scaled as A(i, j) * CTO/CFROM.
      &li_buflen,       // M: No. of rows of the matrix. Treat as a long array.    ,
      &li_one,          // N: No. of columns of the matrix. Treat as a long array.
      A_copy.data_ptr,  // A: The data array.
      &li_buflen,       // LDA: Leading dimension of A.
      &info             // INFO: Exit code.
    );
  }

  #pragma omp barrier
  t1 = omp_get_wtime();
  double pre_time = t1 - t0;

  if (verbose) {
    std::cout << "Maximum entry in tensor: " << anrm << std::endl;
    if (lscl) {
      printf("Exit code for DLASCL: %d\n", info);
      double anrm2 = dlange("M", &li_buflen, &li_one, 
                      A_copy.data_ptr, &li_buflen, nullptr);
      std::cout << "Maximum entry in tensor: " << anrm2 << std::endl;
    }
  }

  #pragma omp barrier
  double qr_time = 0.0, brd_time = 0.0, svd_time = 0.0;
  t0 = omp_get_wtime(); 

  // STAGE 1: Compute the singular values, thresholds, and save computations
  // Get dimensions and workspace for paths
  lapack_int m = A_copy.dims[0];
  lapack_int n = A_copy.dims[1];
  
  lapack_int li_opt = 6;
  lapack_int mnthr = ilaenv(&li_opt, "DGESVD", "VV", &m, &n, &li_zero, &li_zero);
  lapack_int minmn = std::min(m, n);

  lapack_int qrwork = 0, brdwork = 0;
  lapack_int dbdsvdwork = 0; // Only for svdvals

  if (m >= n) {
    if (m >= mnthr) { // Path 1 : M >> N (approx M >= 1.6 N)
      // Workspace query for QR stage
      lapack_int qr_info, qr_lwork;
      double qr_wkopt;

      qr_lwork = -1;
      dgeqrf(&m, &n, nullptr, &m, nullptr, &qr_wkopt, &qr_lwork, &qr_info);
      qrwork = (lapack_int) qr_wkopt;

      // Workspace query for bidiagonal reduction stage
      lapack_int brd_info, brd_lwork;
      double brd_wkopt;

      brd_lwork = -1;
      dgebrd(&m, &n, nullptr, &m, nullptr, nullptr, nullptr, nullptr, 
        &brd_wkopt, &brd_lwork, &brd_info);
      brdwork = (lapack_int) brd_wkopt;

      // Optimal work size needed for bidiagonal SVD
      dbdsvdwork  = 4*n;
    } else { // Path 2 : M >= N
      // Workspace query for bidiagonal reduction stage
      lapack_int brd_info, brd_lwork;
      double brd_wkopt;

      brd_lwork = -1;
      dgebrd(&m, &n, nullptr, &m, nullptr, nullptr, nullptr, nullptr, 
        &brd_wkopt, &brd_lwork, &brd_info);
      brdwork = (lapack_int) brd_wkopt;

      // Optimal work size needed for bidiagonal SVD
      dbdsvdwork  = 4*n;
    }
  } else {
    if (n >= mnthr) { // Path 1t : N >> M (approx N >= 1.6 M)
      // Workspace query for QR stage
      lapack_int qr_info, qr_lwork;
      double qr_wkopt;

      qr_lwork = -1;
      dgelqf(&m, &n, nullptr, &m, nullptr, &qr_wkopt, &qr_lwork, &qr_info);
      qrwork = (lapack_int) qr_wkopt;

      // Workspace query for bidiagonal reduction stage
      lapack_int brd_info, brd_lwork;
      double brd_wkopt;

      brd_lwork = -1;
      dgebrd(&m, &n, nullptr, &m, nullptr, nullptr, nullptr, nullptr, 
        &brd_wkopt, &brd_lwork, &brd_info);
      brdwork = (lapack_int) brd_wkopt;

      // Optimal work size needed for bidiagonal SVD
      dbdsvdwork  = 4*m;
    } else { // Path 2t : N >= M
      // Workspace query for bidiagonal reduction stage
      lapack_int brd_info, brd_lwork;
      double brd_wkopt;

      brd_lwork = -1;
      dgebrd(&m, &n, nullptr, &m, nullptr, nullptr, nullptr, nullptr, 
        &brd_wkopt, &brd_lwork, &brd_info);
      brdwork = (lapack_int) brd_wkopt;

      // Optimal work size needed for bidiagonal SVD
      dbdsvdwork  = 4*m;
    }
  }

  if (verbose) {
    std::cout << "Stage 1: Workspace query outputs:" << std::endl;
    std::cout << "QR stage     : " << qrwork << std::endl;
    std::cout << "BRD stage    : " << brdwork << std::endl;
    std::cout << "DBDSVD stage : " << dbdsvdwork << std::endl;
  }

  // Saved temporaries
  double *tau, *taup, *tauq; // Norms of householder reflectors
  double *Tmats;             // Copying over the R/L if a QR is used
  double *Dvecs, *Evecs;     // Diagonal and Sup/subdiagonal vectors

  // Tensor traversal
  size_t slice_size = m*n;            // Strides for a slice of the tensor
  size_t nslices    = A_copy.nslices; // No. of slices of the tensor

  // Full singular values matrix
  Matrix Sfull(minmn*nslices, minmn, nslices);

  // Padding for s vector in DBDSVDX call
  size_t pad = 1;

  // Loop through the slices and compute singular values
  if (m >= n) {
    if (m >= mnthr) { // Path 1
      /*
        A = Q * R = Q * ( QB * B * PB**T )
                  = Q * ( QB * ( UB * S * VB**T ) * PB**T )
        U = Q * QB * UB; V**T = VB**T * PB**T
      */
      //std::cout << "Stage 1: Path 1" << std::endl;

      // QR stage      
      tau = (double*) malloc(minmn * nslices * sizeof(double));

      // BRD stage
      Tmats = (double*) malloc(n * n * nslices * sizeof(double));
      tauq  = (double*) malloc(minmn * nslices * sizeof(double));
      taup  = (double*) malloc(minmn * nslices * sizeof(double));
      Dvecs = (double*) malloc(minmn * nslices * sizeof(double));
      Evecs = (double*) malloc((minmn-1) * nslices * sizeof(double));

      // Set the triangle matrices to zero
      std::memset(Tmats, 0.0, n * n * nslices * sizeof(double));

#pragma omp parallel
      {
        // Capture thread id
        int tid = omp_get_thread_num();

        // Create temporaries
        // QR stage
        double* workqr = (double*) malloc(qrwork * sizeof(double));

        // BRD stage
        double* workbrd  = (double*) malloc(brdwork * sizeof(double));

        // DBSVD stage
        double* workdbdsvd;
        lapack_int* iworkdbdsvd;
        workdbdsvd  = (double*) malloc(dbdsvdwork * sizeof(double));

#pragma omp for
        for (size_t i = 0; i < nslices; i++) {
          // Copy over a slice
          Matrix As = A.getfrontalslice_copy(i);
          A_copy.setfrontalslice(A.getfrontalslice_copy(i), i);
          As.clear();

          //#pragma omp barrier
          double t0_qr = omp_get_wtime();

          // QR stage
          double* slice_loc = A_copy.data_ptr + (i * slice_size);
          double* tau_loc   = tau + (i * minmn);

          // LAPACK variables
          lapack_int info = 0;

          dgeqrf(
            &m,        // M: No. of rows of the slice.
            &n,        // N: No. of columns of the slice.
            slice_loc, // A: Starting of slice data (overwritten with QR stuff).
            &m,        // LDA: Leading dimension of the slice.
            tau_loc,   // TAU: Scalar factors of the reflectors.
            workqr,    // WORK: Scratch space.
            &qrwork,   // LWORK: Dimension of the array WORK.
            &info      // INFO: Exit code.
          );

          //#pragma omp barrier
          double t1_qr = omp_get_wtime();
          if (tid == 0) {
            qr_time += t1_qr - t0_qr;
          }

          //#pragma omp barrier
          double t0_brd = omp_get_wtime();

          // BRD stage
          // Copy over R from the QR stage to a temporary
          double* A_loc = A_copy.data_ptr + (i * slice_size);
          double* T_loc = Tmats + (i * n * n);

          dlacpy(
            "U",    // UPLO: Upper triangular part of A.
            &n,     // M: No. of rows of the matrix A.
            &n,     // N: No. of columns of the matrix A.
            A_loc,  // A: Matrix to copy from.
            &m,     // LDA: Leading dimension of A.
            T_loc,  // B: Matrix to copy to.
            &n      // LDB: Leading dimension of B.
          );

          // Reduce to BRD form
          double* tauq_loc = tauq + (i * minmn);
          double* taup_loc = taup + (i * minmn);
          double* D_loc    = Dvecs + (i * minmn);
          double* E_loc    = Evecs + (i * (minmn-1));

          // LAPACK variables
          info = 0;

          dgebrd(
            &n,       // M: No. of the rows in A.
            &n,       // N: No. of columns in A.
            T_loc,    // A: Matrix to bidiagonalise.
            &n,       // LDA: Leading dimension of A.
            D_loc,    // D: Diagonal elements of B.
            E_loc,    // E: Superdiagonal elements of B.
            tauq_loc, // TAUQ: Scalar factors of the QB reflectors.
            taup_loc, // TAUP: Scalar factors of the PB reflectors.
            workbrd,  // WORK: Scratch space.
            &brdwork, // LWORK: Dimension of the array WORK.
            &info     // INFO: Exit code.
          );

          //#pragma omp barrier
          double t1_brd = omp_get_wtime();
          if (tid == 0) {
            brd_time += t1_brd - t0_brd;
          }

          //std::cout << "Printing Dvecs[" << i << "]" << std::endl;
          //for (size_t ii = 0; ii < minmn; ii++) {
          //  std::cout << Dvecs[i*minmn + ii] << " ";
          //}
          //std::cout << std::endl;

          //std::cout << "Printing Evecs[" << i << "]" << std::endl;
          //for (size_t ii = 0; ii < minmn-1; ii++) {
          //  std::cout << Evecs[i*(minmn-1) + ii] << " ";
          //}
          //std::cout << std::endl;
          //#pragma omp barrier
          double t0_svd = omp_get_wtime();

          // DBDSVD stage
          std::vector<double> s(minmn+pad);
          D_loc       = Dvecs + (i * minmn);
          E_loc       = Evecs + (i * (minmn-1));
          double zero = 0.0;

          // Copying over D and E (destroyed by DBDSQR)
          std::vector<double> s2(minmn-1+pad);
          for (int jj = 0; jj < minmn - 1; jj++) {
            s[jj] = *(D_loc + jj);
            s2[jj] = *(E_loc + jj);
          }
          s[minmn-1] = *(D_loc + minmn - 1);

          // LAPACK variables
          info = 0;
          lapack_int ns = 0;

          dbdsqr(
            "U",        // UPLO: B is upper bidiagonal.
            &n,         // N: Order of the bidiagonal matrix.
            &li_zero,   // NCVT: No. of columns of VT (0 for only singular values).
            &li_zero,   // NRU: No. of rows of U (0 for only singular values).
            &li_zero,   // NCVT: No. of columns of C (0 for only singular values).
            s.data(),   // D: Diagonal elements of B (overwritten with singular values).
            s2.data(),  // E: Superdiagonal elements of B (destroyed if successful).
            nullptr,    // VT: Not referenced.
            &li_one,    // LDVT: Leading dimension of VT.
            nullptr,    // U: Not referenced.
            &li_one,    // LDU: Leading dimension of U.
            nullptr,    // C: Not referenced.
            &li_one,    // LDC: Leading dimension of C.
            workdbdsvd, // WORK: Scratch space.
            &info       // INFO: Exit code.
          );

          // Set the full matrix singular values
          s.resize(minmn);
          Sfull.setcol(s, i);

          //#pragma omp barrier
          double t1_svd = omp_get_wtime();
          if (tid == 0) {
            svd_time += t1_svd - t0_svd;
          }
        }

        // Free temporaries (if any)
        // QR stage
        free(workqr);

        // BRD stage
        free(workbrd);

        // DBSVD stage
        free(workdbdsvd);
      }

      //std::cout << "Printing Dvecs" << std::endl;
      //for (size_t i = 0; i < minmn*nslices; i++) {
      //  if (i % minmn == 0) std::cout << std::endl;
      //  std::cout << Dvecs[i] << " ";
      //}
      //std::cout << std::endl;

      //std::cout << "Printing Evecs" << std::endl;
      //for (size_t i = 0; i < (minmn - 1) *nslices; i++) {
      //  if (i % (minmn - 1) == 0) std::cout << std::endl;
      //  std::cout << Evecs[i] << " ";
      //}
      //std::cout << std::endl;

    } else { // Path 2
      /*
        A = QB * B * PB**T = QB * ( UB * S * VB**T ) * PB**T
        U = QB * UB; V**T = VB**T * PB**T
      */
      //std::cout << "Stage 1: Path 2" << std::endl;

      // BRD stage
      tauq  = (double*) malloc(minmn * nslices * sizeof(double));
      taup  = (double*) malloc(minmn * nslices * sizeof(double));
      Dvecs = (double*) malloc(minmn * nslices * sizeof(double));
      Evecs = (double*) malloc((minmn-1) * nslices * sizeof(double));

#pragma omp parallel
      {
        // Capture thread id
        int tid = omp_get_thread_num();

        // Create temporaries
        // BRD stage
        double* workbrd  = (double*) malloc(brdwork * sizeof(double));

        // DBSVD stage
        double* workdbdsvd;
        lapack_int* iworkdbdsvd;
        workdbdsvd  = (double*) malloc(dbdsvdwork * sizeof(double));

#pragma omp for
        for (size_t i = 0; i < nslices; i++) {
          // Copy over a slice
          Matrix As = A.getfrontalslice_copy(i);
          A_copy.setfrontalslice(A.getfrontalslice_copy(i), i);
          As.clear();

          //#pragma omp barrier
          double t0_brd = omp_get_wtime();

          // BRD stage
          double* A_loc    = A_copy.data_ptr + (i * slice_size);
          double* tauq_loc = tauq + (i * minmn);
          double* taup_loc = taup + (i * minmn);
          double* D_loc    = Dvecs + (i * minmn);
          double* E_loc    = Evecs + (i * (minmn-1));

          // LAPACK variables
          lapack_int info = 0;

          dgebrd(
            &m,       // M: No. of the rows in A.
            &n,       // N: No. of columns in A.
            A_loc,    // A: Matrix to bidiagonalise.
            &m,       // LDA: Leading dimension of A.
            D_loc,    // D: Diagonal elements of B.
            E_loc,    // E: Superdiagonal elements of B.
            tauq_loc, // TAUQ: Scalar factors of the QB reflectors.
            taup_loc, // TAUP: Scalar factors of the PB reflectors.
            workbrd,  // WORK: Scratch space.
            &brdwork, // LWORK: Dimension of the array WORK.
            &info
          );

          //#pragma omp barrier
          double t1_brd = omp_get_wtime();
          if (tid == 0) {
            brd_time += t1_brd - t0_brd;
          }

          //std::cout << "Printing Dvecs[" << i << "]" << std::endl;
          //for (size_t ii = 0; ii < minmn; ii++) {
          //  std::cout << Dvecs[i*minmn + ii] << " ";
          //}
          //std::cout << std::endl;
          //std::cout << "Printing Evecs[" << i << "]" << std::endl;
          //for (size_t ii = 0; ii < minmn-1; ii++) {
          //  std::cout << Evecs[i*(minmn-1) + ii] << " ";
          //}
          //std::cout << std::endl;

          //#pragma omp barrier
          double t0_svd = omp_get_wtime();

          // DBDSVD stage
          std::vector<double> s(minmn+pad);
          D_loc       = Dvecs + (i * minmn);
          E_loc       = Evecs + (i * (minmn-1));
          double zero = 0.0;

          // Copying over D and E (destroyed by DBDSQR)
          std::vector<double> s2(minmn-1+pad);
          for (int jj = 0; jj < minmn - 1; jj++) {
            s[jj] = *(D_loc + jj);
            s2[jj] = *(E_loc + jj);
          }
          s[minmn-1] = *(D_loc + minmn - 1);

          // LAPACK variables
          info = 0;
          lapack_int ns = 0;

          dbdsqr(
            "U",        // UPLO: B is upper bidiagonal.
            &n,         // N: Order of the bidiagonal matrix.
            &li_zero,   // NCVT: No. of columns of VT (0 for only singular values).
            &li_zero,   // NRU: No. of rows of U (0 for only singular values).
            &li_zero,   // NCVT: No. of columns of C (0 for only singular values).
            s.data(),   // D: Diagonal elements of B (overwritten with singular values).
            s2.data(),  // E: Superdiagonal elements of B (destroyed if successful).
            nullptr,    // VT: Not referenced.
            &li_one,    // LDVT: Leading dimension of VT.
            nullptr,    // U: Not referenced.
            &li_one,    // LDU: Leading dimension of U.
            nullptr,    // C: Not referenced.
            &li_one,    // LDC: Leading dimension of C.
            workdbdsvd, // WORK: Scratch space.
            &info       // INFO: Exit code.
          );

          // Set the full matrix singular values
          s.resize(minmn);
          Sfull.setcol(s, i);

          //#pragma omp barrier
          double t1_svd = omp_get_wtime();
          if (tid == 0) {
            svd_time += t1_svd - t0_svd;
          }
        }
        // Free temporaries (if any)
        // BRD stage
        free(workbrd);

        // DBSVD stage
        free(workdbdsvd);
      }

      //std::cout << "Printing Dvecs" << std::endl;
      //for (size_t i = 0; i < minmn*nslices; i++) {
      //  if (i % minmn == 0) std::cout << std::endl;
      //  std::cout << Dvecs[i] << " ";
      //}
      //std::cout << std::endl;

      //std::cout << "Printing Evecs" << std::endl;
      //for (size_t i = 0; i < (minmn - 1) *nslices; i++) {
      //  if (i % (minmn - 1) == 0) std::cout << std::endl;
      //  std::cout << Evecs[i] << " ";
      //}
      //std::cout << std::endl;
    }
  } else {
    if (n >= mnthr) { // Path 1t
      /*
        A = L * Q = ( QB * B * PB**T ) * Q
                  = ( QB * ( UB * S * VB**T ) * PB**T ) * Q
        U = QB * UB ; V**T = VB**T * PB**T * Q
      */
      //std::cout << "Stage 1: Path 1t" << std::endl;
      
      // LQ stage
      tau = (double*) malloc(minmn * nslices * sizeof(double));

      // BRD stage
      Tmats = (double*) malloc(m * m * nslices * sizeof(double));
      tauq  = (double*) malloc(minmn * nslices * sizeof(double));
      taup  = (double*) malloc(minmn * nslices * sizeof(double));
      Dvecs = (double*) malloc(minmn * nslices * sizeof(double));
      Evecs = (double*) malloc((minmn-1) * nslices * sizeof(double));

      // Set the triangle matrices to zero
      std::memset(Tmats, 0.0, m * m * nslices * sizeof(double));

#pragma omp parallel
      {
        // Capture thread id
        int tid = omp_get_thread_num();

        // Create temporaries
        // LQ stage
        double* workqr = (double*) malloc(qrwork * sizeof(double));

        // BRD stage
        double* workbrd = (double*) malloc(brdwork * sizeof(double));

        // DBSVD stage
        double* workdbdsvd;
        lapack_int* iworkdbdsvd;
        workdbdsvd  = (double*) malloc(dbdsvdwork * sizeof(double));

#pragma omp for
        for (size_t i = 0; i < nslices; i++) {
          // Copy over a slice
          Matrix As = A.getfrontalslice_copy(i);
          A_copy.setfrontalslice(A.getfrontalslice_copy(i), i);
          As.clear();

          //#pragma omp barrier
          double t0_qr = omp_get_wtime();

          // LQ stage
          double* slice_loc = A_copy.data_ptr + (i * slice_size);
          double* tau_loc   = tau + (i * minmn);

          // LAPACK variables
          lapack_int info = 0;
  
          dgelqf(
            &m,        // M: No. of rows of the slice.
            &n,        // N: No. of columns of the slice.
            slice_loc, // A: Starting of slice data (overwritten with QR stuff).
            &m,        // LDA: Leading dimension of the slice.
            tau_loc,   // TAU: Scalar factors of the reflectors.
            workqr,    // WORK: Scratch space.
            &qrwork,   // LWORK: Dimension of the array WORK.
            &info      // INFO: Exit code.
          );

          //#pragma omp barrier
          double t1_qr = omp_get_wtime();
          if (tid == 0) {
            qr_time += t1_qr - t0_qr;
          }

          //#pragma omp barrier
          double t0_brd = omp_get_wtime();

          // BRD stage
          // Copy over L from the LQ stage to a temporary
          double* A_loc = A_copy.data_ptr + (i * slice_size);
          double* T_loc = Tmats + (i * m * m);

          dlacpy(
            "L",    // UPLO: Lower triangular part of A.
            &m,     // M: No. of rows of the matrix A.
            &m,     // N: No. of columns of the matrix A.
            A_loc,  // A: Matrix to copy from.
            &m,     // LDA: Leading dimension of A.
            T_loc,  // B: Matrix to copy to.
            &m      // LDB: Leading dimension of B.
          );

          // Reduce to BRD form
          double* tauq_loc = tauq + (i * minmn);
          double* taup_loc = taup + (i * minmn);
          double* D_loc    = Dvecs + (i * minmn);
          double* E_loc    = Evecs + (i * (minmn-1));

          // LAPACK variables
          info = 0;

          dgebrd(
            &m,       // M: No. of the rows in A.
            &m,       // N: No. of columns in A.
            T_loc,    // A: Matrix to bidiagonalise.
            &m,       // LDA: Leading dimension of A.
            D_loc,    // D: Diagonal elements of B.
            E_loc,    // E: Superdiagonal elements of B.
            tauq_loc, // TAUQ: Scalar factors of the QB reflectors.
            taup_loc, // TAUP: Scalar factors of the PB reflectors.
            workbrd,  // WORK: Scratch space.
            &brdwork, // LWORK: Dimension of the array WORK.
            &info
          );

          //#pragma omp barrier
          double t1_brd = omp_get_wtime();
          if (tid == 0) {
            brd_time += t1_brd - t0_brd;
          }

          //std::cout << "Printing Dvecs[" << i << "]" << std::endl;
          //for (size_t ii = 0; ii < minmn; ii++) {
          //  std::cout << Dvecs[i*minmn + ii] << " ";
          //}
          //std::cout << std::endl;
          //std::cout << "Printing Evecs[" << i << "]" << std::endl;
          //for (size_t ii = 0; ii < minmn-1; ii++) {
          //  std::cout << Evecs[i*(minmn-1) + ii] << " ";
          //}
          //std::cout << std::endl;
          //#pragma omp barrier
          double t0_svd = omp_get_wtime();

          // DBDSVD stage
          std::vector<double> s(minmn+pad);
          D_loc       = Dvecs + (i * minmn);
          E_loc       = Evecs + (i * (minmn-1));
          double zero = 0.0;

          // Copying over D and E (destroyed by DBDSQR)
          std::vector<double> s2(minmn-1+pad);
          for (int jj = 0; jj < minmn - 1; jj++) {
            s[jj] = *(D_loc + jj);
            s2[jj] = *(E_loc + jj);
          }
          s[minmn-1] = *(D_loc + minmn - 1);

          // LAPACK variables
          info = 0;
          lapack_int ns = 0;

          dbdsqr(
            "U",        // UPLO: B is upper bidiagonal.
            &m,         // N: Order of the bidiagonal matrix.
            &li_zero,   // NCVT: No. of columns of VT (0 for only singular values).
            &li_zero,   // NRU: No. of rows of U (0 for only singular values).
            &li_zero,   // NCVT: No. of columns of C (0 for only singular values).
            s.data(),   // D: Diagonal elements of B (overwritten with singular values).
            s2.data(),  // E: Superdiagonal elements of B (destroyed if successful).
            nullptr,    // VT: Not referenced.
            &li_one,    // LDVT: Leading dimension of VT.
            nullptr,    // U: Not referenced.
            &li_one,    // LDU: Leading dimension of U.
            nullptr,    // C: Not referenced.
            &li_one,    // LDC: Leading dimension of C.
            workdbdsvd, // WORK: Scratch space.
            &info       // INFO: Exit code.
          );

          // Set the full matrix singular values
          s.resize(minmn);
          Sfull.setcol(s, i);

          //#pragma omp barrier
          double t1_svd = omp_get_wtime();
          if (tid == 0) {
            svd_time += t1_svd - t0_svd;
          }
        }
        // Free temporaries (if any)
        // LQ stage
        free(workqr);

        // BRD stage
        free(workbrd);

        // DBSVD stage
        free(workdbdsvd);
      }

      //std::cout << "Printing Dvecs" << std::endl;
      //for (size_t i = 0; i < minmn*nslices; i++) {
      //  if (i % minmn == 0) std::cout << std::endl;
      //  std::cout << Dvecs[i] << " ";
      //}
      //std::cout << std::endl;

      //std::cout << "Printing Evecs" << std::endl;
      //for (size_t i = 0; i < (minmn - 1) *nslices; i++) {
      //  if (i % (minmn - 1) == 0) std::cout << std::endl;
      //  std::cout << Evecs[i] << " ";
      //}
      //std::cout << std::endl;

    } else { // Path 2t
      /*
        A = QB * B * PB**T = QB * ( UB * S * VB**T ) * PB**T
        U = QB * UB; V**T = VB**T * PB**T
      */
      //std::cout << "Stage 1: Path 2t" << std::endl;

      // BRD stage
      tauq  = (double*) malloc(minmn * nslices * sizeof(double));
      taup  = (double*) malloc(minmn * nslices * sizeof(double));
      Dvecs = (double*) malloc(minmn * nslices * sizeof(double));
      Evecs = (double*) malloc((minmn-1) * nslices * sizeof(double));

#pragma omp parallel
      {
        // Capture thread id
        int tid = omp_get_thread_num();

        // Create temporaries
        // BRD stage
        double* workbrd = (double*) malloc(brdwork * sizeof(double));

        // DBSVD stage
        double* workdbdsvd;
        lapack_int* iworkdbdsvd;
        workdbdsvd  = (double*) malloc(dbdsvdwork * sizeof(double));

#pragma omp for
        for (size_t i = 0; i < nslices; i++) {
          // Copy over a slice
          Matrix As = A.getfrontalslice_copy(i);
          A_copy.setfrontalslice(A.getfrontalslice_copy(i), i);
          As.clear();

          //#pragma omp barrier
          double t0_brd = omp_get_wtime();

          // BRD stage
          double* A_loc = A_copy.data_ptr + (i * slice_size);
          double* tauq_loc = tauq + (i * minmn);
          double* taup_loc = taup + (i * minmn);
          double* D_loc    = Dvecs + (i * minmn);
          double* E_loc    = Evecs + (i * (minmn-1));

          // LAPACK variables
          lapack_int info = 0;

          dgebrd(
            &m,       // M: No. of the rows in A.
            &n,       // N: No. of columns in A.
            A_loc,    // A: Matrix to bidiagonalise.
            &m,       // LDA: Leading dimension of A.
            D_loc,    // D: Diagonal elements of B.
            E_loc,    // E: Superdiagonal elements of B.
            tauq_loc, // TAUQ: Scalar factors of the QB reflectors.
            taup_loc, // TAUP: Scalar factors of the PB reflectors.
            workbrd,  // WORK: Scratch space.
            &brdwork, // LWORK: Dimension of the array WORK.
            &info
          );

          //#pragma omp barrier
          double t1_brd = omp_get_wtime();
          if (tid == 0) {
            brd_time += t1_brd - t0_brd;
          }

          //std::cout << "Printing Dvecs[" << i << "]" << std::endl;
          //for (size_t ii = 0; ii < minmn; ii++) {
          //  std::cout << Dvecs[i*minmn + ii] << " ";
          //}
          //std::cout << std::endl;
          //std::cout << "Printing Evecs[" << i << "]" << std::endl;
          //for (size_t ii = 0; ii < minmn-1; ii++) {
          //  std::cout << Evecs[i*(minmn-1) + ii] << " ";
          //}
          //std::cout << std::endl;
          //#pragma omp barrier
          double t0_svd = omp_get_wtime();

          // DBDSVD stage
          std::vector<double> s(minmn+pad);
          D_loc       = Dvecs + (i * minmn);
          E_loc       = Evecs + (i * (minmn-1));
          double zero = 0.0;

          // Copying over D and E (destroyed by DBDSQR)
          std::vector<double> s2(minmn-1+pad);
          for (int jj = 0; jj < minmn - 1; jj++) {
            s[jj] = *(D_loc + jj);
            s2[jj] = *(E_loc + jj);
          }
          s[minmn-1] = *(D_loc + minmn - 1);

          // LAPACK variables
          info = 0;
          lapack_int ns = 0;

          dbdsqr(
            "L",        // UPLO: B is upper bidiagonal.
            &m,         // N: Order of the bidiagonal matrix.
            &li_zero,   // NCVT: No. of columns of VT (0 for only singular values).
            &li_zero,   // NRU: No. of rows of U (0 for only singular values).
            &li_zero,   // NCVT: No. of columns of C (0 for only singular values).
            s.data(),   // D: Diagonal elements of B (overwritten with singular values).
            s2.data(),  // E: Subdiagonal elements of B (destroyed if successful).
            nullptr,    // VT: Not referenced.
            &li_one,    // LDVT: Leading dimension of VT.
            nullptr,    // U: Not referenced.
            &li_one,    // LDU: Leading dimension of U.
            nullptr,    // C: Not referenced.
            &li_one,    // LDC: Leading dimension of C.
            workdbdsvd, // WORK: Scratch space.
            &info       // INFO: Exit code.
          );

          // Set the full matrix singular values
          s.resize(minmn);
          Sfull.setcol(s, i);

          //#pragma omp barrier
          double t1_svd = omp_get_wtime();
          if (tid == 0) {
            svd_time += t1_svd - t0_svd;
          }
        }
        // Free temporaries (if any)
        // BRD stage
        free(workbrd);

        // DBSVD stage
        free(workdbdsvd);
      }

      //std::cout << "Printing Dvecs" << std::endl;
      //for (size_t i = 0; i < minmn*nslices; i++) {
      //  if (i % minmn == 0) std::cout << std::endl;
      //  std::cout << Dvecs[i] << " ";
      //}
      //std::cout << std::endl;

      //std::cout << "Printing Evecs" << std::endl;
      //for (size_t i = 0; i < (minmn - 1) *nslices; i++) {
      //  if (i % (minmn - 1) == 0) std::cout << std::endl;
      //  std::cout << Evecs[i] << " ";
      //}
      //std::cout << std::endl;
    }
  }

  //std::cout << "Singular values computed." << std::endl;
  //Sfull.print();
  #pragma omp barrier
  t1 = omp_get_wtime();
  double stage1_time = t1 - t0;

  #pragma omp barrier
  t0 = omp_get_wtime();
  // Compute the thresholds
  std::vector<size_t> ks = thresholds(Sfull, tol);
  if (verbose) {
    std::cout << "Ranks found." << std::endl;
    for (size_t ii = 0; ii < ks.size(); ii++) {
      std::cout << ks[ii] << " ";
    }
    std::cout << std::endl;
  }
  Sfull.clear();
  #pragma omp barrier
  t1 = omp_get_wtime();
  double thr_time = t1 - t0;
  
  #pragma omp barrier
  t0 = omp_get_wtime();
  // STAGE 2: Recompute singular values and vectors with correct ranks

  // Create the output variables
  size_t U_fixed_dim = m;
  size_t U_buflen    = 0;
  for (size_t r = 0; r < ks.size(); r++) {
    U_buflen += U_fixed_dim * ks[r];
  }
  JaggedTensor U(U_buflen, U_fixed_dim, ks);

  size_t Vt_fixed_dim = n;
  size_t Vt_buflen    = 0;
  for (size_t r = 0; r < ks.size(); r++) {
    Vt_buflen += Vt_fixed_dim * ks[r];
  }
  JaggedTensor Vt(Vt_buflen, Vt_fixed_dim, ks, true);

  size_t S_buflen = std::accumulate(ks.begin(), ks.end(), (size_t) 0);
  JaggedMatrix S(S_buflen, ks);

  // Get dimensions and workspaces for paths
  lapack_int qrwork2 = 0, pbrwork = 0, qbrwork = 0;
  lapack_int dbdsvdwork2 = 0, dbdsvdiwork2 = 0, dbdsvdzwork = 0;

  // Find the largest rank
  size_t kmax        = *std::max_element(ks.begin(), ks.end());
  lapack_int li_kmax = kmax;

  if (m >= n) {
    if (m >= mnthr) { // Path 1: M >> N (approx M >= 1.6 N)
      // Optimal work size needed for bidiagonal SVD
      dbdsvdwork2  = 14*n;
      dbdsvdiwork2 = 12*n;
      dbdsvdzwork  = li_kmax*(n*2+1);

      // Workspace query for bidiagonal reduction stage (max work)
      lapack_int br_info, br_lwork;
      double br_wkopt;
      
      br_lwork = -1;
      dormbr("Q", "L", "N", &n, &li_kmax, &n, nullptr, &n, nullptr, 
        nullptr, &m, &br_wkopt, &br_lwork, &br_info);
      qbrwork = (lapack_int) br_wkopt;

      br_lwork = -1;
      dormbr("P", "R", "T", &li_kmax, &n, &n, nullptr, &n, nullptr, 
        nullptr, &li_kmax, &br_wkopt, &br_lwork, &br_info);
      pbrwork = (lapack_int) br_wkopt;

      // Workspace query for QR stage
      lapack_int qr_info, qr_lwork;
      double qr_wkopt;
      
      qr_lwork = -1;
      dormqr("L", "N", &m, &li_kmax, &n, nullptr, &m, nullptr, nullptr,
        &m, &qr_wkopt, &qr_lwork, &qr_info);
      qrwork2 = (lapack_int) qr_wkopt;
    } else { // Path 2 : M >= N
      // Optimal work size needed for bidiagonal SVD
      dbdsvdwork2  = 14*n;
      dbdsvdiwork2 = 12*n;
      dbdsvdzwork  = li_kmax*(n*2+1);

      // Workspace query for bidiagonal reduction stage (max work)
      lapack_int br_info, br_lwork;
      double br_wkopt;
      
      br_lwork = -1;
      dormbr("Q", "L", "N", &m, &li_kmax, &n, nullptr, &m, nullptr, 
        nullptr, &m, &br_wkopt, &br_lwork, &br_info);
      qbrwork = (lapack_int) br_wkopt;

      br_lwork = -1;
      dormbr("P", "R", "T", &li_kmax, &n, &m, nullptr, &m, nullptr, 
        nullptr, &li_kmax, &br_wkopt, &br_lwork, &br_info);
      pbrwork = (lapack_int) br_wkopt;
    }
  } else { 
    if (n >= mnthr) { // Path 1t: N >> M (approx N >= 1.6 M)
      // Optimal work size needed for bidiagonal SVD
      dbdsvdwork2  = 14*m;
      dbdsvdiwork2 = 12*m;
      dbdsvdzwork  = li_kmax*(m*2+1);

      // Workspace query for bidiagonal reduction stage (max work)
      lapack_int br_info, br_lwork;
      double br_wkopt;
      
      br_lwork = -1;
      dormbr("Q", "L", "N", &m, &li_kmax, &m, nullptr, &m, nullptr, 
        nullptr, &m, &br_wkopt, &br_lwork, &br_info);
      qbrwork = (lapack_int) br_wkopt;

      br_lwork = -1;
      dormbr("P", "R", "T", &li_kmax, &m, &m, nullptr, &m, nullptr, 
        nullptr, &li_kmax, &br_wkopt, &br_lwork, &br_info);
      pbrwork = (lapack_int) br_wkopt;

      // Workspace query for QR stage
      lapack_int qr_info, qr_lwork;
      double qr_wkopt;

      qr_lwork = -1;
      dormlq("R", "N", &li_kmax, &n, &m, nullptr, &m, nullptr, nullptr,
        &li_kmax, &qr_wkopt, &qr_lwork, &qr_info);
      qrwork2 = (lapack_int) qr_wkopt;
    } else { // Path 2t : N >= M
      // Optimal work size needed for bidiagonal SVD
      dbdsvdwork2  = 14*m;
      dbdsvdiwork2 = 12*m;
      dbdsvdzwork  = li_kmax*(m*2+1);

      // Workspace query for bidiagonal reduction stage (max work)
      lapack_int br_info, br_lwork;
      double br_wkopt;
      
      br_lwork = -1;
      dormbr("Q", "L", "N", &m, &li_kmax, &n, nullptr, &m, nullptr, 
        nullptr, &m, &br_wkopt, &br_lwork, &br_info);
      qbrwork = (lapack_int) br_wkopt;

      br_lwork = -1;
      dormbr("P", "R", "T", &li_kmax, &n, &m, nullptr, &m, nullptr, 
        nullptr, &li_kmax, &br_wkopt, &br_lwork, &br_info);
      pbrwork = (lapack_int) br_wkopt;
    }
  }

  if (verbose) {
    std::cout << "Stage 2: Workspace query outputs:" << std::endl;
    std::cout << "DBDSVD stage : " << dbdsvdwork2 << " " 
              << dbdsvdiwork2 << " " << dbdsvdzwork << std::endl;
    std::cout << "BRD stage    : " << qbrwork << " " << pbrwork << std::endl;
    std::cout << "QR stage     : " << qrwork2 << std::endl;
  }

  // Loop through the slices and compute the truncated SVD
  if (m >= n) {
    if (m >= mnthr) { // Path 1
      /*
        A = Q * R = Q * ( QB * B * PB**T )
                  = Q * ( QB * ( UB * S * VB**T ) * PB**T )
        U = Q * QB * UB; V**T = VB**T * PB**T
      */
      //std::cout << "Stage 2: Path 1" << std::endl;

#pragma omp parallel
      {
        // Create temporaries
        // DBDSVD stage
        double *workdbdsvd, *workdbdsvdz;
        lapack_int* iworkdbdsvd;
        workdbdsvd  = (double*) malloc(dbdsvdwork2 * sizeof(double));
        workdbdsvdz = (double*) malloc(dbdsvdzwork * sizeof(double));
        iworkdbdsvd = (lapack_int*) malloc(dbdsvdiwork2 * sizeof(lapack_int));

        // Computing U stage
        double *workqbr = (double*) malloc(qbrwork * sizeof(double));
        double *workqr2 = (double*) malloc(qrwork2 * sizeof(double));

        // Computing Vt stage
        double *workpbr = (double*) malloc(pbrwork * sizeof(double));

#pragma omp for
        for (size_t i = 0; i < nslices; i++) {
          if (ks[i] > 0) {
            // Inputs for DBDSVD
            //std::cout << "Printing Dvecs[" << i << "]" << std::endl;
            //for (size_t ii = 0; ii < minmn; ii++) {
            //  std::cout << Dvecs[i*minmn + ii] << " ";
            //}
            //std::cout << std::endl;

            //std::cout << "Printing Evecs[" << i << "]" << std::endl;
            //for (size_t ii = 0; ii < minmn-1; ii++) {
            //  std::cout << Evecs[i*(minmn-1) + ii] << " ";
            //}
            //std::cout << std::endl;
            
            // Outputs for this slice
            size_t k = ks[i];
            Matrix Uk(m*k, m, k);
            Matrix Vkt(k*n, k, n);
            std::vector<double> s(minmn+pad);
            
            lapack_int ldu = m, ldvt = k;

            // Ensure matrices are zero initialised
            std::memset(Uk.data_ptr, 0.0, m * k * sizeof(double));
            std::memset(Vkt.data_ptr, 0.0, k * n * sizeof(double));

            // DBDSVD stage (recalculate the singular values and vectors)
            double *D_loc = Dvecs + (i * minmn);
            double *E_loc = Evecs + (i * (minmn - 1));
            double zero   = 0.0;
            
            // LAPACK variables
            lapack_int info = 0;
            lapack_int ns = 0, il = 1, iu = k, ldz = 2*n;

            dbdsvdx(
              "U",         // UPLO: B is upper bidiagonal.
              "V",         // JOBZ: Compute singular values and vectors.
              "I",         // RANGE: Compute singular values in index range.
              &n,          // N: Order of the bidiagonal matrix.
              D_loc,       // D: Diagonal elements of B.
              E_loc,       // E: Superdiagonal elements of B.
              &zero,       // VL: Not referenced.
              &zero,       // VU: Not referenced.
              &il,         // IL: Lower index of singular values returned. 1 <= IL <= IU <= min(M, N).
              &iu,         // IU: Upper index of singular values returned. 1 <= IL <= IU <= min(M, N).
              &ns,         // NS: No. of singular values found.
              s.data(),    // S: Array holding the singular values.
              workdbdsvdz, // Z: Array containing the singular vectors.
              &ldz,        // LDZ: Leading dimension of Z.
              workdbdsvd,  // WORK: Scratch space.
              iworkdbdsvd, // IWORK: Integer work containing indices of unconverged elements on failure.
              &info        // INFO: Exit code.
            );

            // Resize the singular values to k
            s.resize(k);

            // Copy over UB and VB**T
            for (size_t jj = 0; jj < k; jj++) {
              // Go through the Z array column by column
              for (size_t ii = 0; ii < n; ii++) {
                Uk.set(ii, jj, workdbdsvdz[(jj * (2 * n)) + ii]);
                Vkt.set(jj, ii, workdbdsvdz[(jj * (2 * n)) + n + ii]);
              }
            }
          
            // Compute the left singular vectors : U = Q * QB * UB
            double *T_loc    = Tmats + (i * n * n);
            double *tauq_loc = tauq + (i * minmn);

            // LAPACK variables
            lapack_int nk = k;
            info = 0;
           
            // Compute QB * UB
            // Here C = UB
            dormbr(
              "Q",         // VECT: Applying a QB matrix to UB.
              "L",         // SIDE: From the left.
              "N",         // TRANS: Not transposed.
              &n,          // M: No. of rows of the matrix UB.
              &nk,         // N: No. of columns of the matrix UB.
              &n,          // K: No. of columns in matrix reduced by DGEBRD.
              T_loc,       // A: Matrix overwritten by DGEBRD.
              &n,          // LDA: Leading dimension of A.
              tauq_loc,    // TAU: Scalar factors of the QB reflector.
              Uk.data_ptr, // C: The UB matrix containing the left singular vectors.
              &ldu,        // LDC: Leading dimension of UB.
              workqbr,     // WORK: Scratch space.
              &qbrwork,    // LWORK: Dimension of the array WORK.
              &info        // INFO: Exit code.
            );

            // Compute Q * QB * UB
            double *A_loc   = A_copy.data_ptr + (i * slice_size);
            double *tau_loc = tau + (i * minmn);

            // LAPACK variables
            info = 0;

            dormqr(
              "L",         // SIDE: Apply a Q matrix from the left.
              "N",         // TRANS: Q is not transposed.
              &m,          // M: No. of rows QB * UB.
              &nk,         // N: No. of columns of QB * UB.
              &n,          // K: No. of elementary reflectors in Q.
              A_loc,       // A: Matrix overwritten by DGEQRF.
              &m,          // LDA: Leading dimension of A.
              tau_loc,     // TAU: Scalar factors of the Q reflector.
              Uk.data_ptr, // C: The QB * UB matrix containing the left singular vectors.
              &ldu,        // LDC: Leading dimension of C.
              workqr2,     // WORK: Scratch space.
              &qrwork2,    // LWORK: Dimension of the array WORK.
              &info        // INFO: Exit code. 
            );

            // Compute the right singular vectors: V**T = VB**T * PB**T
            double *taup_loc = taup + (i * minmn);

            // LAPACK variables
            nk = k;
            info = 0;
           
            // Compute VB**T * PB**T
            // Here C = VB**T
            dormbr(
              "P",          // VECT: Applying a PB**T matrix to VB**T.
              "R",          // SIDE: From the right.
              "T",          // TRANS: Transposed.
              &nk,          // M: No. of rows of the matrix VB**T.
              &n,           // N: No. of columns of the matrix VB**T.
              &n,           // K: No. of rows in matrix reduced by DGEBRD.
              T_loc,        // A: Matrix overwritten by DGEBRD.
              &n,           // LDA: Leading dimension of A.
              taup_loc,     // TAU: Scalar factors of the PB reflector.
              Vkt.data_ptr, // C: The UB matrix containing the left singular vectors.
              &ldvt,        // LDC: Leading dimension of UB.
              workpbr,      // WORK: Scratch space.
              &pbrwork,     // LWORK: Dimension of the array WORK.
              &info         // INFO: Exit code.
            );

            // Save the output
            U.setfrontalslice(Uk, i);
            S.setcol(s, i);
            Vt.setfrontalslice(Vkt, i);

            // Clear the temporaries
            Uk.clear();
            Vkt.clear();
          }
        }
        // Free temporaries (if any)
        // DBDSVD stage
        free(workdbdsvdz);
        free(workdbdsvd);
        free(iworkdbdsvd);

        // Computing U
        free(workqbr);
        free(workqr2);

        // Computing Vt
        free(workpbr);
      }

      // Free saved computations
      // QR stage
      free(tau);

      // BRD stage
      free(Tmats);
      free(tauq);
      free(taup);
      free(Dvecs);
      free(Evecs);

    } else { // Path 2
      /*
        A = QB * B * PB**T = QB * ( UB * S * VB**T ) * PB**T
        U = QB * UB; V**T = VB**T * PB**T
      */
      //std::cout << "Stage 2: Path 2" << std::endl;

#pragma omp parallel
      {
        // Create temporaries
        double *workdbdsvd, *workdbdsvdz;
        lapack_int* iworkdbdsvd;
        workdbdsvd  = (double*) malloc(dbdsvdwork2 * sizeof(double));
        workdbdsvdz = (double*) malloc(dbdsvdzwork * sizeof(double));
        iworkdbdsvd = (lapack_int*) malloc(dbdsvdiwork2 * sizeof(lapack_int));

        // Computing U stage
        double *workqbr = (double*) malloc(qbrwork * sizeof(double));

        // Computing Vt stage
        double *workpbr = (double*) malloc(pbrwork * sizeof(double));

#pragma omp for
        for (size_t i = 0; i < nslices; i++) {
          if (ks[i] > 0) {
            // Inputs for DBDSVD
            //std::cout << "Printing Dvecs[" << i << "]" << std::endl;
            //for (size_t ii = 0; ii < minmn; ii++) {
            //  std::cout << Dvecs[i*minmn + ii] << " ";
            //}
            //std::cout << std::endl;

            //std::cout << "Printing Evecs[" << i << "]" << std::endl;
            //for (size_t ii = 0; ii < minmn-1; ii++) {
            //  std::cout << Evecs[i*(minmn-1) + ii] << " ";
            //}
            //std::cout << std::endl;
            
            // Outputs for this slice
            size_t k = ks[i];
            Matrix Uk(m*k, m, k);
            Matrix Vkt(k*n, k, n);
            std::vector<double> s(minmn+pad);
            
            lapack_int ldu = m, ldvt = k;

            // Ensure matrices are zero initialised
            std::memset(Uk.data_ptr, 0.0, m * k * sizeof(double));
            std::memset(Vkt.data_ptr, 0.0, k * n * sizeof(double));

            // DBDSVD stage (recalculate the singular values and vectors)
            double *D_loc = Dvecs + (i * minmn);
            double *E_loc = Evecs + (i * (minmn - 1));
            double zero   = 0.0;
            
            // LAPACK variables
            lapack_int info = 0;
            lapack_int ns = 0, il = 1, iu = k, ldz = 2*n;

            dbdsvdx(
              "U",         // UPLO: B is upper bidiagonal.
              "V",         // JOBZ: Compute singular values and vectors.
              "I",         // RANGE: Compute singular values in index range.
              &n,          // N: Order of the bidiagonal matrix.
              D_loc,       // D: Diagonal elements of B.
              E_loc,       // E: Superdiagonal elements of B.
              &zero,       // VL: Not referenced.
              &zero,       // VU: Not referenced.
              &il,         // IL: Lower index of singular values returned. 1 <= IL <= IU <= min(M, N).
              &iu,         // IU: Upper index of singular values returned. 1 <= IL <= IU <= min(M, N).
              &ns,         // NS: No. of singular values found.
              s.data(),    // S: Array holding the singular values.
              workdbdsvdz, // Z: Array containing the singular vectors.
              &ldz,        // LDZ: Leading dimension of Z.
              workdbdsvd,  // WORK: Scratch space.
              iworkdbdsvd, // IWORK: Integer work containing indices of unconverged elements on failure.
              &info        // INFO: Exit code.
            );

            // Resize the singular values to k
            s.resize(k);

            // Copy over UB and VB**T
            for (size_t jj = 0; jj < k; jj++) {
              // Go through the Z array column by column
              for (size_t ii = 0; ii < n; ii++) {
                Uk.set(ii, jj, workdbdsvdz[(jj * (2 * n)) + ii]);
                Vkt.set(jj, ii, workdbdsvdz[(jj * (2 * n)) + n + ii]);
              }
            }
            
            // Compute the left singular vectors : U = Q * QB * UB
            double *A_loc   = A_copy.data_ptr + (i * slice_size);
            double *tauq_loc = tauq + (i * minmn);

            // LAPACK variables
            lapack_int nk = k;
            info = 0;
           
            // Compute QB * UB
            // Here C = UB
            dormbr(
              "Q",         // VECT: Applying a QB matrix to UB.
              "L",         // SIDE: From the left.
              "N",         // TRANS: Not transposed.
              &m,          // M: No. of rows of the matrix UB.
              &nk,         // N: No. of columns of the matrix UB.
              &n,          // K: No. of columns in matrix reduced by DGEBRD.
              A_loc,       // A: Matrix overwritten by DGEBRD.
              &m,          // LDA: Leading dimension of A.
              tauq_loc,    // TAU: Scalar factors of the QB reflector.
              Uk.data_ptr, // C: The UB matrix containing the left singular vectors.
              &ldu,        // LDC: Leading dimension of UB.
              workqbr,     // WORK: Scratch space.
              &qbrwork,    // LWORK: Dimension of the array WORK.
              &info        // INFO: Exit code.
            );

            // Compute the right singular vectors: V**T = VB**T * PB**T
            double *taup_loc = taup + (i * minmn);

            // LAPACK variables
            nk = k;
            info = 0;
           
            // Compute VB**T * PB**T
            // Here C = VB**T
            dormbr(
              "P",          // VECT: Applying a PB**T matrix to VB**T.
              "R",          // SIDE: From the right.
              "T",          // TRANS: Transposed.
              &nk,          // M: No. of rows of the matrix VB**T.
              &n,           // N: No. of columns of the matrix VB**T.
              &m,           // K: No. of rows in matrix reduced by DGEBRD.
              A_loc,        // A: Matrix overwritten by DGEBRD.
              &m,           // LDA: Leading dimension of A.
              taup_loc,     // TAU: Scalar factors of the PB reflector.
              Vkt.data_ptr, // C: The UB matrix containing the left singular vectors.
              &ldvt,        // LDC: Leading dimension of UB.
              workpbr,      // WORK: Scratch space.
              &pbrwork,     // LWORK: Dimension of the array WORK.
              &info         // INFO: Exit code.
            );

            // Save the output
            U.setfrontalslice(Uk, i);
            S.setcol(s, i);
            Vt.setfrontalslice(Vkt, i);

            // Clear the temporaries
            Uk.clear();
            Vkt.clear();
          }
        }
        // Free temporaries (if any)
        // DBDSVD stage
        free(workdbdsvdz);
        free(workdbdsvd);
        free(iworkdbdsvd);

        // Computing U
        free(workqbr);

        // Computing Vt
        free(workpbr);
      }
      // Free saved computations
      // BRD stage
      free(tauq);
      free(taup);
      free(Dvecs);
      free(Evecs);
    }
  } else {
    if (n >= mnthr) { // Path 1t
      /*
        A = L * Q = ( QB * B * PB**T ) * Q
                  = ( QB * ( UB * S * VB**T ) * PB**T ) * Q
        U = QB * UB ; V**T = VB**T * PB**T * Q
      */
      //std::cout << "Stage 2: Path 1t" << std::endl;

#pragma omp parallel
      {
        // Create temporaries
        // DBDSVD stage
        double *workdbdsvd, *workdbdsvdz;
        lapack_int* iworkdbdsvd;
        workdbdsvd  = (double*) malloc(dbdsvdwork2 * sizeof(double));
        workdbdsvdz = (double*) malloc(dbdsvdzwork * sizeof(double));
        iworkdbdsvd = (lapack_int*) malloc(dbdsvdiwork2 * sizeof(lapack_int));

        // Computing U stage
        double *workqbr = (double*) malloc(qbrwork * sizeof(double));

        // Computing Vt stage
        double *workpbr = (double*) malloc(pbrwork * sizeof(double));
        double *workqr2 = (double*) malloc(qrwork2 * sizeof(double));

#pragma omp for
        for (size_t i = 0; i < nslices; i++) {
          if (ks[i] > 0) {
            // Inputs for DBDSVD
            //std::cout << "Printing Dvecs[" << i << "]" << std::endl;
            //for (size_t ii = 0; ii < minmn; ii++) {
            //  std::cout << Dvecs[i*minmn + ii] << " ";
            //}
            //std::cout << std::endl;

            //std::cout << "Printing Evecs[" << i << "]" << std::endl;
            //for (size_t ii = 0; ii < minmn-1; ii++) {
            //  std::cout << Evecs[i*(minmn-1) + ii] << " ";
            //}
            //std::cout << std::endl;
            
            // Outputs for this slice
            size_t k = ks[i];
            Matrix Uk(m*k, m, k);
            Matrix Vkt(k*n, k, n);
            std::vector<double> s(minmn+pad);
            
            lapack_int ldu = m, ldvt = k;

            // Ensure matrices are zero initialised
            std::memset(Uk.data_ptr, 0.0, m * k * sizeof(double));
            std::memset(Vkt.data_ptr, 0.0, k * n * sizeof(double));

            // DBDSVD stage (recalculate the singular values and vectors)
            double *D_loc = Dvecs + (i * minmn);
            double *E_loc = Evecs + (i * (minmn - 1));
            double zero   = 0.0;
            
            // LAPACK variables
            lapack_int info = 0;
            lapack_int ns = 0, il = 1, iu = k, ldz = 2*m;

            dbdsvdx(
              "U",         // UPLO: B is upper bidiagonal.
              "V",         // JOBZ: Compute singular values and vectors.
              "I",         // RANGE: Compute singular values in index range.
              &m,          // N: Order of the bidiagonal matrix.
              D_loc,       // D: Diagonal elements of B.
              E_loc,       // E: Superdiagonal elements of B.
              &zero,       // VL: Not referenced.
              &zero,       // VU: Not referenced.
              &il,         // IL: Lower index of singular values returned. 1 <= IL <= IU <= min(M, N).
              &iu,         // IU: Upper index of singular values returned. 1 <= IL <= IU <= min(M, N).
              &ns,         // NS: No. of singular values found.
              s.data(),    // S: Array holding the singular values.
              workdbdsvdz, // Z: Array containing the singular vectors.
              &ldz,        // LDZ: Leading dimension of Z.
              workdbdsvd,  // WORK: Scratch space.
              iworkdbdsvd, // IWORK: Integer work containing indices of unconverged elements on failure.
              &info        // INFO: Exit code.
            );

            // Resize the singular values to k
            s.resize(k);

            // Copy over UB and VB**T
            for (size_t jj = 0; jj < k; jj++) {
              // Go through the Z array column by column
              for (size_t ii = 0; ii < m; ii++) {
                Uk.set(ii, jj, workdbdsvdz[(jj * (2 * m)) + ii]);
                Vkt.set(jj, ii, workdbdsvdz[(jj * (2 * m)) + m + ii]);
              }
            }
          
            // Compute the left singular vectors : U = QB * UB
            double *T_loc    = Tmats + (i * m * m);
            double *tauq_loc = tauq + (i * minmn);

            // LAPACK variables
            lapack_int nk = k;
            info = 0;
           
            // Compute QB * UB
            // Here C = UB
            dormbr(
              "Q",         // VECT: Applying a QB matrix to UB.
              "L",         // SIDE: From the left.
              "N",         // TRANS: Not transposed.
              &m,          // M: No. of rows of the matrix UB.
              &nk,         // N: No. of columns of the matrix UB.
              &m,          // K: No. of columns in matrix reduced by DGEBRD.
              T_loc,       // A: Matrix overwritten by DGEBRD.
              &m,          // LDA: Leading dimension of A.
              tauq_loc,    // TAU: Scalar factors of the QB reflector.
              Uk.data_ptr, // C: The UB matrix containing the left singular vectors.
              &ldu,        // LDC: Leading dimension of UB.
              workqbr,     // WORK: Scratch space.
              &qbrwork,    // LWORK: Dimension of the array WORK.
              &info        // INFO: Exit code.
            );

            // Compute the right singular vectors: V**T = VB**T * PB**T * Q
            double *taup_loc = taup + (i * minmn);

            // LAPACK variables
            nk = k;
            info = 0;
           
            // Compute VB**T * PB**T
            // Here C = VB**T
            dormbr(
              "P",          // VECT: Applying a PB**T matrix to VB**T.
              "R",          // SIDE: From the right.
              "T",          // TRANS: Transposed.
              &nk,          // M: No. of rows of the matrix VB**T.
              &m,           // N: No. of columns of the matrix VB**T.
              &m,           // K: No. of rows in matrix reduced by DGEBRD.
              T_loc,        // A: Matrix overwritten by DGEBRD.
              &m,           // LDA: Leading dimension of A.
              taup_loc,     // TAU: Scalar factors of the PB reflector.
              Vkt.data_ptr, // C: The VB**T matrix containing the right singular vectors.
              &ldvt,        // LDC: Leading dimension of C.
              workpbr,      // WORK: Scratch space.
              &pbrwork,     // LWORK: Dimension of the array WORK.
              &info         // INFO: Exit code.
            );

            // Compute VB**T * PB**T * Q
            double *A_loc   = A_copy.data_ptr + (i * slice_size);
            double *tau_loc = tau + (i * minmn);

            // LAPACK variables
            info = 0;

            // Vkt is k x n and is C in this context
            // C = [VB**T * PB**T 0]
            dormlq(
              "R",          // SIDE: Apply a Q matrix to the right.
              "N",          // TRANS: Q is not transposed.
              &nk,          // M: No. of rows of the matrix C.
              &n,           // N: No. of columns of the matrix C.
              &m,           // K: No. of elementary reflectors in Q.
              A_loc,        // A: Matrix overwritten by DGELQF.
              &m,           // LDA: Leading dimension of A.
              tau_loc,      // TAU: Scalar factors of the Q reflector.
              Vkt.data_ptr, // C: The Vkt matrix containing the right singular vectors.
              &ldvt,        // LDC: Leading dimension of C.
              workqr2,      // WORK: Scratch space.
              &qrwork2,     // LWORK: Dimension of the array WORK.
              &info         // INFO: Exit code.
            );

            // Save the output
            U.setfrontalslice(Uk, i);
            S.setcol(s, i);
            Vt.setfrontalslice(Vkt, i);

            // Clear the temporaries
            Uk.clear();
            Vkt.clear();
          }
        }
        // Free temporaries (if any)
        // DBDSVD stage
        free(workdbdsvdz);
        free(workdbdsvd);
        free(iworkdbdsvd);

        // Computing U
        free(workqbr);

        // Computing Vt
        free(workpbr);
        free(workqr2);
      }
      // Free saved computations
      // QR stage
      free(tau);

      // BRD stage
      free(Tmats);
      free(tauq);
      free(taup);
      free(Dvecs);
      free(Evecs);
    } else { // Path 2t
      /*
        A = QB * B * PB**T = QB * ( UB * S * VB**T ) * PB**T
        U = QB * UB; V**T = VB**T * PB**T
      */
      //std::cout << "Stage 2: Path 2t" << std::endl;

#pragma omp parallel
      {
        // Create temporaries
        // DBDSVD stage
        double *workdbdsvd, *workdbdsvdz;
        lapack_int* iworkdbdsvd;
        workdbdsvd  = (double*) malloc(dbdsvdwork2 * sizeof(double));
        workdbdsvdz = (double*) malloc(dbdsvdzwork * sizeof(double));
        iworkdbdsvd = (lapack_int*) malloc(dbdsvdiwork2 * sizeof(lapack_int));

        // Computing U stage
        double *workqbr = (double*) malloc(qbrwork * sizeof(double));

        // Computing Vt stage
        double *workpbr = (double*) malloc(pbrwork * sizeof(double));

#pragma omp for
        for (size_t i = 0; i < nslices; i++) {
          if (ks[i] > 0) {
            // Inputs for DBDSVD
            //std::cout << "Printing Dvecs[" << i << "]" << std::endl;
            //for (size_t ii = 0; ii < minmn; ii++) {
            //  std::cout << Dvecs[i*minmn + ii] << " ";
            //}
            //std::cout << std::endl;

            //std::cout << "Printing Evecs[" << i << "]" << std::endl;
            //for (size_t ii = 0; ii < minmn-1; ii++) {
            //  std::cout << Evecs[i*(minmn-1) + ii] << " ";
            //}
            //std::cout << std::endl;
            
            // Outputs for this slice
            size_t k = ks[i];
            Matrix Uk(m*k, m, k);
            Matrix Vkt(k*n, k, n);
            std::vector<double> s(minmn+pad);
            
            lapack_int ldu = m, ldvt = k;

            // Ensure matrices are zero initialised
            std::memset(Uk.data_ptr, 0.0, m * k * sizeof(double));
            std::memset(Vkt.data_ptr, 0.0, k * n * sizeof(double));

            // DBDSVD stage (recalculate the singular values and vectors)
            double *D_loc = Dvecs + (i * minmn);
            double *E_loc = Evecs + (i * (minmn - 1));
            double zero   = 0.0;
            
            // LAPACK variables
            lapack_int info = 0;
            lapack_int ns = 0, il = 1, iu = k, ldz = 2*m;

            dbdsvdx(
              "L",         // UPLO: B is lower bidiagonal.
              "V",         // JOBZ: Compute singular values and vectors.
              "I",         // RANGE: Compute singular values in index range.
              &m,          // N: Order of the bidiagonal matrix.
              D_loc,       // D: Diagonal elements of B.
              E_loc,       // E: Subdiagonal elements of B.
              &zero,       // VL: Not referenced.
              &zero,       // VU: Not referenced.
              &il,         // IL: Lower index of singular values returned. 1 <= IL <= IU <= min(M, N).
              &iu,         // IU: Upper index of singular values returned. 1 <= IL <= IU <= min(M, N).
              &ns,         // NS: No. of singular values found.
              s.data(),    // S: Array holding the singular values.
              workdbdsvdz, // Z: Array containing the singular vectors.
              &ldz,        // LDZ: Leading dimension of Z.
              workdbdsvd,  // WORK: Scratch space.
              iworkdbdsvd, // IWORK: Integer work containing indices of unconverged elements on failure.
              &info        // INFO: Exit code.
            );

            // Resize the singular values to k
            s.resize(k);

            // Copy over UB and VB**T
            for (size_t jj = 0; jj < k; jj++) {
              // Go through the Z array column by column
              for (size_t ii = 0; ii < m; ii++) {
                Uk.set(ii, jj, workdbdsvdz[(jj * (2 * m)) + ii]);
                Vkt.set(jj, ii, workdbdsvdz[(jj * (2 * m)) + m + ii]);
              }
            }
          
            // Compute the left singular vectors : U = QB * UB
            double *A_loc    = A_copy.data_ptr + (i * slice_size);
            double *tauq_loc = tauq + (i * minmn);

            // LAPACK variables
            lapack_int nk = k;
            info = 0;
           
            // Compute QB * UB
            // Here C = UB
            dormbr(
              "Q",         // VECT: Applying a QB matrix to UB.
              "L",         // SIDE: From the left.
              "N",         // TRANS: Not transposed.
              &m,          // M: No. of rows of the matrix UB.
              &nk,         // N: No. of columns of the matrix UB.
              &n,          // K: No. of columns in matrix reduced by DGEBRD.
              A_loc,       // A: Matrix overwritten by DGEBRD.
              &m,          // LDA: Leading dimension of A.
              tauq_loc,    // TAU: Scalar factors of the QB reflector.
              Uk.data_ptr, // C: The UB matrix containing the left singular vectors.
              &ldu,        // LDC: Leading dimension of UB.
              workqbr,     // WORK: Scratch space.
              &qbrwork,    // LWORK: Dimension of the array WORK.
              &info        // INFO: Exit code.
            );

            // Compute the right singular vectors: V**T = VB**T * PB**T * Q
            double *taup_loc = taup + (i * minmn);

            // LAPACK variables
            nk = k;
            info = 0;
           
            // Compute VB**T * PB**T
            // Here C = VB**T
            dormbr(
              "P",          // VECT: Applying a PB**T matrix to VB**T.
              "R",          // SIDE: From the right.
              "T",          // TRANS: Transposed.
              &nk,          // M: No. of rows of the matrix VB**T.
              &n,           // N: No. of columns of the matrix VB**T.
              &m,           // K: No. of rows in matrix reduced by DGEBRD.
              A_loc,        // A: Matrix overwritten by DGEBRD.
              &m,           // LDA: Leading dimension of A.
              taup_loc,     // TAU: Scalar factors of the PB reflector.
              Vkt.data_ptr, // C: The VB**T matrix containing the right singular vectors.
              &ldvt,        // LDC: Leading dimension of C.
              workpbr,      // WORK: Scratch space.
              &pbrwork,     // LWORK: Dimension of the array WORK.
              &info         // INFO: Exit code.
            );

            // Save the output
            U.setfrontalslice(Uk, i);
            S.setcol(s, i);
            Vt.setfrontalslice(Vkt, i);

            // Clear the temporaries
            Uk.clear();
            Vkt.clear();
          }
        }
        // Free temporaries (if any)
        // DBDSVD stage
        free(workdbdsvdz);
        free(workdbdsvd);
        free(iworkdbdsvd);

        // Computing U
        free(workqbr);

        // Computing Vt
        free(workpbr);
      }
      // Free saved computations
      // BRD stage
      free(tauq);
      free(taup);
      free(Dvecs);
      free(Evecs);
    }
  }
  #pragma omp barrier
  t1 = omp_get_wtime();
  double stage2_time = t1 - t0;

  #pragma omp barrier
  t0 = omp_get_wtime();

  // Undo scaling
  if (lscl) {
    lapack_int li_sbuflen = S_buflen;
    if (anrm > 0.0 && anrm < smlnum) {
      //std::cout << "Reached SMLNUM" << std::endl;
      dlascl(
        "G",         // G: A is a full matrix.
        &li_zero,    // KL: Not referenced.
        &li_zero,    // KU: Not referenced.
        &smlnum,     // CFROM: Matrix scaled as A(i, j) * CTO/CFROM.
        &anrm,       // CTO: Matrix scaled as A(i, j) * CTO/CFROM.
        &li_sbuflen, // M: No. of rows of the matrix. Treat as a long array.
        &li_one,     // N: No. of columns of the matrix. Treat as a long array.
        S.data_ptr,  // A: The data array.
        &li_sbuflen, // LDA: Leading dimension of A.
        &info        // INFO: Exit code.
      );
    } else if (anrm > bignum) {
      //std::cout << "Reached BIGNUM" << std::endl;
      dlascl( 
        "G",         // G: A is a full matrix.
        &li_zero,    // KL: Not referenced.
        &li_zero,    // KU: Not referenced.
        &bignum,     // CFROM: Matrix scaled as A(i, j) * CTO/CFROM.
        &anrm,       // CTO: Matrix scaled as A(i, j) * CTO/CFROM.
        &li_sbuflen, // M: No. of rows of the matrix. Treat as a long array.    ,
        &li_one,     // N: No. of columns of the matrix. Treat as a long array.
        S.data_ptr,  // A: The data array.
        &li_sbuflen, // LDA: Leading dimension of A.
        &info        // INFO: Exit code.
      );
    }
  }

  // Free the copied tensor
  A_copy.clear();

  #pragma omp barrier
  t1 = omp_get_wtime();
  double post_time = t1 - t0;

  std::cout << "Pre time    : " << pre_time << std::endl
            << "Stage 1 time: " << stage1_time << std::endl
            << "  QR time   : " << qr_time << std::endl
            << "  BRD time  : " << brd_time << std::endl
            << "  SVD time  : " << svd_time << std::endl
            << "Thr time    : " << thr_time << std::endl
            << "Stage 2 time: " << stage2_time << std::endl
            << "Post time   : " << post_time << std::endl;

  return std::make_tuple(std::move(U), std::move(S), std::move(Vt));
}

std::tuple<JaggedTensor, JaggedMatrix, JaggedTensor> slicewise_svd_thr2(
  const Tensor &A, double tol, bool verbose=false) {

  #pragma omp barrier
  double t0, t1;
  t0 = omp_get_wtime();

  // STAGE 0: Preprocessing
  Tensor A_copy(A); // call a copy tensor

  // Scale the data
  // Get the machine constants
  double eps = dlamch("P");
  double smlnum = std::sqrt(dlamch("S")) / eps;
  double bignum = 1.0 / smlnum;

  //std::cout << eps << " " << smlnum << " " << bignum << std::endl;
  // Scale A if max element outside range [smlnum, bignum]
  lapack_int li_buflen = A_copy.buflen;
  lapack_int li_one = 1, li_zero = 0;
  lapack_int info = 0;
  // Get max(abs(a(i,j)))
  double anrm = dlange("M", &li_buflen, &li_one, 
                  A_copy.data_ptr, &li_buflen, nullptr);

  bool lscl = false;
  if (anrm > 0.0 && anrm < smlnum) {
    lscl = true;
    //std::cout << "Reached SMLNUM" << std::endl;
    dlascl(
      "G",              // G: A is a full matrix.
      &li_zero,         // KL: Not referenced.
      &li_zero,         // KU: Not referenced.
      &anrm,            // CFROM: Matrix scaled as A(i, j) * CTO/CFROM.
      &smlnum,          // CTO: Matrix scaled as A(i, j) * CTO/CFROM.
      &li_buflen,       // M: No. of rows of the matrix. Treat as a long array.
      &li_one,          // N: No. of columns of the matrix. Treat as a long array.
      A_copy.data_ptr,  // A: The data array.
      &li_buflen,       // LDA: Leading dimension of A.
      &info             // INFO: Exit code.
    );
  } else if (anrm > bignum) {
    lscl = true;
    //std::cout << "Reached BIGNUM" << std::endl;
    dlascl( 
      "G",              // G: A is a full matrix.
      &li_zero,         // KL: Not referenced.
      &li_zero,         // KU: Not referenced.
      &anrm,            // CFROM: Matrix scaled as A(i, j) * CTO/CFROM.
      &bignum,          // CTO: Matrix scaled as A(i, j) * CTO/CFROM.
      &li_buflen,       // M: No. of rows of the matrix. Treat as a long array.    ,
      &li_one,          // N: No. of columns of the matrix. Treat as a long array.
      A_copy.data_ptr,  // A: The data array.
      &li_buflen,       // LDA: Leading dimension of A.
      &info             // INFO: Exit code.
    );
  }

  #pragma omp barrier
  t1 = omp_get_wtime();
  double pre_time = t1 - t0;

  if (verbose) {
    std::cout << "Maximum entry in tensor: " << anrm << std::endl;
    if (lscl) {
      printf("Exit code for DLASCL: %d\n", info);
      double anrm2 = dlange("M", &li_buflen, &li_one, 
                      A_copy.data_ptr, &li_buflen, nullptr);
      std::cout << "Maximum entry in tensor: " << anrm2 << std::endl;
    }
  }

  #pragma omp barrier
  double qr_time = 0.0, brd_time = 0.0, svd_time = 0.0;
  t0 = omp_get_wtime(); 

  // STAGE 1: Compute the singular values, thresholds, and save computations
  // Get dimensions and workspace for paths
  lapack_int m = A_copy.dims[0];
  lapack_int n = A_copy.dims[1];
  
  lapack_int li_opt = 6;
  lapack_int mnthr = ilaenv(&li_opt, "DGESVD", "VV", &m, &n, &li_zero, &li_zero);
  lapack_int minmn = std::min(m, n);

  lapack_int qrwork = 0, brdwork = 0;
  lapack_int dbdsvdwork = 0, dbdsvdiwork = 0; // Only for svdvals

  if (m >= n) {
    if (m >= mnthr) { // Path 1 : M >> N (approx M >= 1.6 N)
      // Workspace query for QR stage
      lapack_int qr_info, qr_lwork;
      double qr_wkopt;

      qr_lwork = -1;
      dgeqrf(&m, &n, nullptr, &m, nullptr, &qr_wkopt, &qr_lwork, &qr_info);
      qrwork = (lapack_int) qr_wkopt;

      // Workspace query for bidiagonal reduction stage
      lapack_int brd_info, brd_lwork;
      double brd_wkopt;

      brd_lwork = -1;
      dgebrd(&m, &n, nullptr, &m, nullptr, nullptr, nullptr, nullptr, 
        &brd_wkopt, &brd_lwork, &brd_info);
      brdwork = (lapack_int) brd_wkopt;

      // Optimal work size needed for bidiagonal SVD
      dbdsvdwork  = 14*n;
      dbdsvdiwork = 12*n;
    } else { // Path 2 : M >= N
      // Workspace query for bidiagonal reduction stage
      lapack_int brd_info, brd_lwork;
      double brd_wkopt;

      brd_lwork = -1;
      dgebrd(&m, &n, nullptr, &m, nullptr, nullptr, nullptr, nullptr, 
        &brd_wkopt, &brd_lwork, &brd_info);
      brdwork = (lapack_int) brd_wkopt;

      // Optimal work size needed for bidiagonal SVD
      dbdsvdwork  = 14*n;
      dbdsvdiwork = 12*n;
    }
  } else {
    if (n >= mnthr) { // Path 1t : N >> M (approx N >= 1.6 M)
      // Workspace query for QR stage
      lapack_int qr_info, qr_lwork;
      double qr_wkopt;

      qr_lwork = -1;
      dgelqf(&m, &n, nullptr, &m, nullptr, &qr_wkopt, &qr_lwork, &qr_info);
      qrwork = (lapack_int) qr_wkopt;

      // Workspace query for bidiagonal reduction stage
      lapack_int brd_info, brd_lwork;
      double brd_wkopt;

      brd_lwork = -1;
      dgebrd(&m, &n, nullptr, &m, nullptr, nullptr, nullptr, nullptr, 
        &brd_wkopt, &brd_lwork, &brd_info);
      brdwork = (lapack_int) brd_wkopt;

      // Optimal work size needed for bidiagonal SVD
      dbdsvdwork  = 14*m;
      dbdsvdiwork = 12*m;
    } else { // Path 2t : N >= M
      // Workspace query for bidiagonal reduction stage
      lapack_int brd_info, brd_lwork;
      double brd_wkopt;

      brd_lwork = -1;
      dgebrd(&m, &n, nullptr, &m, nullptr, nullptr, nullptr, nullptr, 
        &brd_wkopt, &brd_lwork, &brd_info);
      brdwork = (lapack_int) brd_wkopt;

      // Optimal work size needed for bidiagonal SVD
      dbdsvdwork  = 14*m;
      dbdsvdiwork = 12*m;
    }
  }

  if (verbose) {
    std::cout << "Stage 1: Workspace query outputs:" << std::endl;
    std::cout << "QR stage     : " << qrwork << std::endl;
    std::cout << "BRD stage    : " << brdwork << std::endl;
    std::cout << "DBDSVD stage : " << dbdsvdwork << " " << dbdsvdiwork << std::endl;
  }

  // Saved temporaries
  double *tau, *taup, *tauq; // Norms of householder reflectors
  double *Tmats;             // Copying over the R/L if a QR is used
  double *Dvecs, *Evecs;     // Diagonal and Sup/subdiagonal vectors

  // Tensor traversal
  size_t slice_size = m*n;            // Strides for a slice of the tensor
  size_t nslices    = A_copy.nslices; // No. of slices of the tensor

  // Full singular values matrix
  Matrix Sfull(minmn*nslices, minmn, nslices);

  // Padding for s vector in DBDSVDX call
  size_t pad = 1;

  // Loop through the slices and compute singular values
  if (m >= n) {
    if (m >= mnthr) { // Path 1
      /*
        A = Q * R = Q * ( QB * B * PB**T )
                  = Q * ( QB * ( UB * S * VB**T ) * PB**T )
        U = Q * QB * UB; V**T = VB**T * PB**T
      */
      //std::cout << "Stage 1: Path 1" << std::endl;

      // QR stage      
      tau = (double*) malloc(minmn * nslices * sizeof(double));

      // BRD stage
      Tmats = (double*) malloc(n * n * nslices * sizeof(double));
      tauq  = (double*) malloc(minmn * nslices * sizeof(double));
      taup  = (double*) malloc(minmn * nslices * sizeof(double));
      Dvecs = (double*) malloc(minmn * nslices * sizeof(double));
      Evecs = (double*) malloc((minmn-1) * nslices * sizeof(double));

      // Set the triangle matrices to zero
      std::memset(Tmats, 0.0, n * n * nslices * sizeof(double));

#pragma omp parallel
      {
        // Capture thread id
        int tid = omp_get_thread_num();

        // Create temporaries
        // QR stage
        double* workqr = (double*) malloc(qrwork * sizeof(double));

        // BRD stage
        double* workbrd  = (double*) malloc(brdwork * sizeof(double));

        // DBSVD stage
        double* workdbdsvd;
        lapack_int* iworkdbdsvd;
        workdbdsvd  = (double*) malloc(dbdsvdwork * sizeof(double));
        iworkdbdsvd = (lapack_int*) malloc(dbdsvdiwork * sizeof(lapack_int));

#pragma omp for
        for (size_t i = 0; i < nslices; i++) {
          //#pragma omp barrier
          double t0_qr = omp_get_wtime();

          // QR stage
          double* slice_loc = A_copy.data_ptr + (i * slice_size);
          double* tau_loc   = tau + (i * minmn);

          // LAPACK variables
          lapack_int info = 0;

          dgeqrf(
            &m,        // M: No. of rows of the slice.
            &n,        // N: No. of columns of the slice.
            slice_loc, // A: Starting of slice data (overwritten with QR stuff).
            &m,        // LDA: Leading dimension of the slice.
            tau_loc,   // TAU: Scalar factors of the reflectors.
            workqr,    // WORK: Scratch space.
            &qrwork,   // LWORK: Dimension of the array WORK.
            &info      // INFO: Exit code.
          );

          //#pragma omp barrier
          double t1_qr = omp_get_wtime();
          if (tid == 0) {
            qr_time += t1_qr - t0_qr;
          }

          //#pragma omp barrier
          double t0_brd = omp_get_wtime();

          // BRD stage
          // Copy over R from the QR stage to a temporary
          double* A_loc = A_copy.data_ptr + (i * slice_size);
          double* T_loc = Tmats + (i * n * n);

          dlacpy(
            "U",    // UPLO: Upper triangular part of A.
            &n,     // M: No. of rows of the matrix A.
            &n,     // N: No. of columns of the matrix A.
            A_loc,  // A: Matrix to copy from.
            &m,     // LDA: Leading dimension of A.
            T_loc,  // B: Matrix to copy to.
            &n      // LDB: Leading dimension of B.
          );

          // Reduce to BRD form
          double* tauq_loc = tauq + (i * minmn);
          double* taup_loc = taup + (i * minmn);
          double* D_loc    = Dvecs + (i * minmn);
          double* E_loc    = Evecs + (i * (minmn-1));

          // LAPACK variables
          info = 0;

          dgebrd(
            &n,       // M: No. of the rows in A.
            &n,       // N: No. of columns in A.
            T_loc,    // A: Matrix to bidiagonalise.
            &n,       // LDA: Leading dimension of A.
            D_loc,    // D: Diagonal elements of B.
            E_loc,    // E: Superdiagonal elements of B.
            tauq_loc, // TAUQ: Scalar factors of the QB reflectors.
            taup_loc, // TAUP: Scalar factors of the PB reflectors.
            workbrd,  // WORK: Scratch space.
            &brdwork, // LWORK: Dimension of the array WORK.
            &info     // INFO: Exit code.
          );

          //#pragma omp barrier
          double t1_brd = omp_get_wtime();
          if (tid == 0) {
            brd_time += t1_brd - t0_brd;
          }

          //std::cout << "Printing Dvecs[" << i << "]" << std::endl;
          //for (size_t ii = 0; ii < minmn; ii++) {
          //  std::cout << Dvecs[i*minmn + ii] << " ";
          //}
          //std::cout << std::endl;

          //std::cout << "Printing Evecs[" << i << "]" << std::endl;
          //for (size_t ii = 0; ii < minmn-1; ii++) {
          //  std::cout << Evecs[i*(minmn-1) + ii] << " ";
          //}
          //std::cout << std::endl;
          //#pragma omp barrier
          double t0_svd = omp_get_wtime();

          // DBDSVD stage
          std::vector<double> s(minmn+pad);
          D_loc       = Dvecs + (i * minmn);
          E_loc       = Evecs + (i * (minmn-1));
          double zero = 0.0;

          // LAPACK variables
          info = 0;
          lapack_int ns = 0;

          dbdsvdx(
            "U",         // UPLO: B is upper bidiagonal.
            "N",         // JOBZ: Compute singular values only.
            "A",         // RANGE: Compute all singular values.
            &n,          // N: Order of the bidiagonal matrix.
            D_loc,       // D: Diagonal elements of B.
            E_loc,       // E: Superdiagonal elements of B.
            &zero,       // VL: Not referenced.
            &zero,       // VU: Not referenced.
            &li_zero,    // IL: Not referenced.
            &li_zero,    // IU: Not referenced.
            &ns,         // NS: No. of singular values found.
            s.data(),    // S: Array holding the singular values.
            nullptr,     // Z: Not referenced.
            &li_one,     // LDZ: Leading dimension of Z.
            workdbdsvd,  // WORK: Scratch space.
            iworkdbdsvd, // IWORK: Integer work containing indices of unconverged elements on failure.
            &info        // INFO: Exit code.
          );

          // Set the full matrix singular values
          s.resize(minmn);
          Sfull.setcol(s, i);

          //#pragma omp barrier
          double t1_svd = omp_get_wtime();
          if (tid == 0) {
            svd_time += t1_svd - t0_svd;
          }
        }

        // Free temporaries (if any)
        // QR stage
        free(workqr);

        // BRD stage
        free(workbrd);

        // DBSVD stage
        free(workdbdsvd);
        free(iworkdbdsvd);
      }

      //std::cout << "Printing Dvecs" << std::endl;
      //for (size_t i = 0; i < minmn*nslices; i++) {
      //  if (i % minmn == 0) std::cout << std::endl;
      //  std::cout << Dvecs[i] << " ";
      //}
      //std::cout << std::endl;

      //std::cout << "Printing Evecs" << std::endl;
      //for (size_t i = 0; i < (minmn - 1) *nslices; i++) {
      //  if (i % (minmn - 1) == 0) std::cout << std::endl;
      //  std::cout << Evecs[i] << " ";
      //}
      //std::cout << std::endl;

    } else { // Path 2
      /*
        A = QB * B * PB**T = QB * ( UB * S * VB**T ) * PB**T
        U = QB * UB; V**T = VB**T * PB**T
      */
      //std::cout << "Stage 1: Path 2" << std::endl;

      // BRD stage
      tauq  = (double*) malloc(minmn * nslices * sizeof(double));
      taup  = (double*) malloc(minmn * nslices * sizeof(double));
      Dvecs = (double*) malloc(minmn * nslices * sizeof(double));
      Evecs = (double*) malloc((minmn-1) * nslices * sizeof(double));

#pragma omp parallel
      {
        // Capture thread id
        int tid = omp_get_thread_num();

        // Create temporaries
        // BRD stage
        double* workbrd  = (double*) malloc(brdwork * sizeof(double));

        // DBSVD stage
        double* workdbdsvd;
        lapack_int* iworkdbdsvd;
        workdbdsvd  = (double*) malloc(dbdsvdwork * sizeof(double));
        iworkdbdsvd = (lapack_int*) malloc(dbdsvdiwork * sizeof(lapack_int));

#pragma omp for
        for (size_t i = 0; i < nslices; i++) {
          //#pragma omp barrier
          double t0_brd = omp_get_wtime();

          // BRD stage
          double* A_loc    = A_copy.data_ptr + (i * slice_size);
          double* tauq_loc = tauq + (i * minmn);
          double* taup_loc = taup + (i * minmn);
          double* D_loc    = Dvecs + (i * minmn);
          double* E_loc    = Evecs + (i * (minmn-1));

          // LAPACK variables
          lapack_int info = 0;

          dgebrd(
            &m,       // M: No. of the rows in A.
            &n,       // N: No. of columns in A.
            A_loc,    // A: Matrix to bidiagonalise.
            &m,       // LDA: Leading dimension of A.
            D_loc,    // D: Diagonal elements of B.
            E_loc,    // E: Superdiagonal elements of B.
            tauq_loc, // TAUQ: Scalar factors of the QB reflectors.
            taup_loc, // TAUP: Scalar factors of the PB reflectors.
            workbrd,  // WORK: Scratch space.
            &brdwork, // LWORK: Dimension of the array WORK.
            &info
          );

          //#pragma omp barrier
          double t1_brd = omp_get_wtime();
          if (tid == 0) {
            brd_time += t1_brd - t0_brd;
          }

          //std::cout << "Printing Dvecs[" << i << "]" << std::endl;
          //for (size_t ii = 0; ii < minmn; ii++) {
          //  std::cout << Dvecs[i*minmn + ii] << " ";
          //}
          //std::cout << std::endl;
          //std::cout << "Printing Evecs[" << i << "]" << std::endl;
          //for (size_t ii = 0; ii < minmn-1; ii++) {
          //  std::cout << Evecs[i*(minmn-1) + ii] << " ";
          //}
          //std::cout << std::endl;

          //#pragma omp barrier
          double t0_svd = omp_get_wtime();

          // DBDSVD stage
          std::vector<double> s(minmn+pad);
          D_loc       = Dvecs + (i * minmn);
          E_loc       = Evecs + (i * (minmn-1));
          double zero = 0.0;

          // LAPACK variables
          info = 0;
          lapack_int ns = 0;

          dbdsvdx(
            "U",         // UPLO: B is upper bidiagonal.
            "N",         // JOBZ: Compute singular values only.
            "A",         // RANGE: Compute all singular values.
            &n,          // N: Order of the bidiagonal matrix.
            D_loc,       // D: Diagonal elements of B.
            E_loc,       // E: Superdiagonal elements of B.
            &zero,       // VL: Not referenced.
            &zero,       // VU: Not referenced.
            &li_zero,    // IL: Not referenced.
            &li_zero,    // IU: Not referenced.
            &ns,         // NS: No. of singular values found.
            s.data(),    // S: Array holding the singular values.
            nullptr,     // Z: Not referenced.
            &li_one,     // LDZ: Leading dimension of Z.
            workdbdsvd,  // WORK: Scratch space.
            iworkdbdsvd, // IWORK: Integer work containing indices of unconverged elements on failure.
            &info        // INFO: Exit code.
          );

          // Set the full matrix singular values
          s.resize(minmn);
          Sfull.setcol(s, i);

          //#pragma omp barrier
          double t1_svd = omp_get_wtime();
          if (tid == 0) {
            svd_time += t1_svd - t0_svd;
          }
        }
        // Free temporaries (if any)
        // BRD stage
        free(workbrd);

        // DBSVD stage
        free(workdbdsvd);
        free(iworkdbdsvd);
      }

      //std::cout << "Printing Dvecs" << std::endl;
      //for (size_t i = 0; i < minmn*nslices; i++) {
      //  if (i % minmn == 0) std::cout << std::endl;
      //  std::cout << Dvecs[i] << " ";
      //}
      //std::cout << std::endl;

      //std::cout << "Printing Evecs" << std::endl;
      //for (size_t i = 0; i < (minmn - 1) *nslices; i++) {
      //  if (i % (minmn - 1) == 0) std::cout << std::endl;
      //  std::cout << Evecs[i] << " ";
      //}
      //std::cout << std::endl;
    }
  } else {
    if (n >= mnthr) { // Path 1t
      /*
        A = L * Q = ( QB * B * PB**T ) * Q
                  = ( QB * ( UB * S * VB**T ) * PB**T ) * Q
        U = QB * UB ; V**T = VB**T * PB**T * Q
      */
      //std::cout << "Stage 1: Path 1t" << std::endl;
      
      // LQ stage
      tau = (double*) malloc(minmn * nslices * sizeof(double));

      // BRD stage
      Tmats = (double*) malloc(m * m * nslices * sizeof(double));
      tauq  = (double*) malloc(minmn * nslices * sizeof(double));
      taup  = (double*) malloc(minmn * nslices * sizeof(double));
      Dvecs = (double*) malloc(minmn * nslices * sizeof(double));
      Evecs = (double*) malloc((minmn-1) * nslices * sizeof(double));

      // Set the triangle matrices to zero
      std::memset(Tmats, 0.0, m * m * nslices * sizeof(double));

#pragma omp parallel
      {
        // Capture thread id
        int tid = omp_get_thread_num();

        // Create temporaries
        // LQ stage
        double* workqr = (double*) malloc(qrwork * sizeof(double));

        // BRD stage
        double* workbrd = (double*) malloc(brdwork * sizeof(double));

        // DBSVD stage
        double* workdbdsvd;
        lapack_int* iworkdbdsvd;
        workdbdsvd  = (double*) malloc(dbdsvdwork * sizeof(double));
        iworkdbdsvd = (lapack_int*) malloc(dbdsvdiwork * sizeof(lapack_int));

#pragma omp for
        for (size_t i = 0; i < nslices; i++) {
          //#pragma omp barrier
          double t0_qr = omp_get_wtime();

          // LQ stage
          double* slice_loc = A_copy.data_ptr + (i * slice_size);
          double* tau_loc   = tau + (i * minmn);

          // LAPACK variables
          lapack_int info = 0;
  
          dgelqf(
            &m,        // M: No. of rows of the slice.
            &n,        // N: No. of columns of the slice.
            slice_loc, // A: Starting of slice data (overwritten with QR stuff).
            &m,        // LDA: Leading dimension of the slice.
            tau_loc,   // TAU: Scalar factors of the reflectors.
            workqr,    // WORK: Scratch space.
            &qrwork,   // LWORK: Dimension of the array WORK.
            &info      // INFO: Exit code.
          );

          //#pragma omp barrier
          double t1_qr = omp_get_wtime();
          if (tid == 0) {
            qr_time += t1_qr - t0_qr;
          }

          //#pragma omp barrier
          double t0_brd = omp_get_wtime();

          // BRD stage
          // Copy over L from the LQ stage to a temporary
          double* A_loc = A_copy.data_ptr + (i * slice_size);
          double* T_loc = Tmats + (i * m * m);

          dlacpy(
            "L",    // UPLO: Lower triangular part of A.
            &m,     // M: No. of rows of the matrix A.
            &m,     // N: No. of columns of the matrix A.
            A_loc,  // A: Matrix to copy from.
            &m,     // LDA: Leading dimension of A.
            T_loc,  // B: Matrix to copy to.
            &m      // LDB: Leading dimension of B.
          );

          // Reduce to BRD form
          double* tauq_loc = tauq + (i * minmn);
          double* taup_loc = taup + (i * minmn);
          double* D_loc    = Dvecs + (i * minmn);
          double* E_loc    = Evecs + (i * (minmn-1));

          // LAPACK variables
          info = 0;

          dgebrd(
            &m,       // M: No. of the rows in A.
            &m,       // N: No. of columns in A.
            T_loc,    // A: Matrix to bidiagonalise.
            &m,       // LDA: Leading dimension of A.
            D_loc,    // D: Diagonal elements of B.
            E_loc,    // E: Superdiagonal elements of B.
            tauq_loc, // TAUQ: Scalar factors of the QB reflectors.
            taup_loc, // TAUP: Scalar factors of the PB reflectors.
            workbrd,  // WORK: Scratch space.
            &brdwork, // LWORK: Dimension of the array WORK.
            &info
          );

          //#pragma omp barrier
          double t1_brd = omp_get_wtime();
          if (tid == 0) {
            brd_time += t1_brd - t0_brd;
          }

          //std::cout << "Printing Dvecs[" << i << "]" << std::endl;
          //for (size_t ii = 0; ii < minmn; ii++) {
          //  std::cout << Dvecs[i*minmn + ii] << " ";
          //}
          //std::cout << std::endl;
          //std::cout << "Printing Evecs[" << i << "]" << std::endl;
          //for (size_t ii = 0; ii < minmn-1; ii++) {
          //  std::cout << Evecs[i*(minmn-1) + ii] << " ";
          //}
          //std::cout << std::endl;
          //#pragma omp barrier
          double t0_svd = omp_get_wtime();

          // DBDSVD stage
          std::vector<double> s(minmn+pad);
          D_loc       = Dvecs + (i * minmn);
          E_loc       = Evecs + (i * (minmn-1));
          double zero = 0.0;

          // LAPACK variables
          info = 0;
          lapack_int ns = 0;

          dbdsvdx(
            "U",         // UPLO: B is upper bidiagonal.
            "N",         // JOBZ: Compute singular values only.
            "A",         // RANGE: Compute all singular values.
            &m,          // N: Order of the bidiagonal matrix.
            D_loc,       // D: Diagonal elements of B.
            E_loc,       // E: Superdiagonal elements of B.
            &zero,       // VL: Not referenced.
            &zero,       // VU: Not referenced.
            &li_zero,    // IL: Not referenced.
            &li_zero,    // IU: Not referenced.
            &ns,         // NS: No. of singular values found.
            s.data(),    // S: Array holding the singular values.
            nullptr,     // Z: Not referenced.
            &li_one,     // LDZ: Leading dimension of Z.
            workdbdsvd,  // WORK: Scratch space.
            iworkdbdsvd, // IWORK: Integer work containing indices of unconverged elements on failure.
            &info        // INFO: Exit code.
          );

          // Set the full matrix singular values
          s.resize(minmn);
          Sfull.setcol(s, i);

          //#pragma omp barrier
          double t1_svd = omp_get_wtime();
          if (tid == 0) {
            svd_time += t1_svd - t0_svd;
          }
        }
        // Free temporaries (if any)
        // LQ stage
        free(workqr);

        // BRD stage
        free(workbrd);

        // DBSVD stage
        free(workdbdsvd);
        free(iworkdbdsvd);
      }

      //std::cout << "Printing Dvecs" << std::endl;
      //for (size_t i = 0; i < minmn*nslices; i++) {
      //  if (i % minmn == 0) std::cout << std::endl;
      //  std::cout << Dvecs[i] << " ";
      //}
      //std::cout << std::endl;

      //std::cout << "Printing Evecs" << std::endl;
      //for (size_t i = 0; i < (minmn - 1) *nslices; i++) {
      //  if (i % (minmn - 1) == 0) std::cout << std::endl;
      //  std::cout << Evecs[i] << " ";
      //}
      //std::cout << std::endl;

    } else { // Path 2t
      /*
        A = QB * B * PB**T = QB * ( UB * S * VB**T ) * PB**T
        U = QB * UB; V**T = VB**T * PB**T
      */
      //std::cout << "Stage 1: Path 2t" << std::endl;

      // BRD stage
      tauq  = (double*) malloc(minmn * nslices * sizeof(double));
      taup  = (double*) malloc(minmn * nslices * sizeof(double));
      Dvecs = (double*) malloc(minmn * nslices * sizeof(double));
      Evecs = (double*) malloc((minmn-1) * nslices * sizeof(double));

#pragma omp parallel
      {
        // Capture thread id
        int tid = omp_get_thread_num();

        // Create temporaries
        // BRD stage
        double* workbrd = (double*) malloc(brdwork * sizeof(double));

        // DBSVD stage
        double* workdbdsvd;
        lapack_int* iworkdbdsvd;
        workdbdsvd  = (double*) malloc(dbdsvdwork * sizeof(double));
        iworkdbdsvd = (lapack_int*) malloc(dbdsvdiwork * sizeof(lapack_int));

#pragma omp for
        for (size_t i = 0; i < nslices; i++) {
          //#pragma omp barrier
          double t0_brd = omp_get_wtime();

          // BRD stage
          double* A_loc = A_copy.data_ptr + (i * slice_size);
          double* tauq_loc = tauq + (i * minmn);
          double* taup_loc = taup + (i * minmn);
          double* D_loc    = Dvecs + (i * minmn);
          double* E_loc    = Evecs + (i * (minmn-1));

          // LAPACK variables
          lapack_int info = 0;

          dgebrd(
            &m,       // M: No. of the rows in A.
            &n,       // N: No. of columns in A.
            A_loc,    // A: Matrix to bidiagonalise.
            &m,       // LDA: Leading dimension of A.
            D_loc,    // D: Diagonal elements of B.
            E_loc,    // E: Superdiagonal elements of B.
            tauq_loc, // TAUQ: Scalar factors of the QB reflectors.
            taup_loc, // TAUP: Scalar factors of the PB reflectors.
            workbrd,  // WORK: Scratch space.
            &brdwork, // LWORK: Dimension of the array WORK.
            &info
          );

          //#pragma omp barrier
          double t1_brd = omp_get_wtime();
          if (tid == 0) {
            brd_time += t1_brd - t0_brd;
          }

          //std::cout << "Printing Dvecs[" << i << "]" << std::endl;
          //for (size_t ii = 0; ii < minmn; ii++) {
          //  std::cout << Dvecs[i*minmn + ii] << " ";
          //}
          //std::cout << std::endl;
          //std::cout << "Printing Evecs[" << i << "]" << std::endl;
          //for (size_t ii = 0; ii < minmn-1; ii++) {
          //  std::cout << Evecs[i*(minmn-1) + ii] << " ";
          //}
          //std::cout << std::endl;
          //#pragma omp barrier
          double t0_svd = omp_get_wtime();

          // DBDSVD stage
          std::vector<double> s(minmn+pad);
          D_loc       = Dvecs + (i * minmn);
          E_loc       = Evecs + (i * (minmn-1));
          double zero = 0.0;

          // LAPACK variables
          info = 0;
          lapack_int ns = 0;

          dbdsvdx(
            "L",         // UPLO: B is lower bidiagonal.
            "N",         // JOBZ: Compute singular values only.
            "A",         // RANGE: Compute all singular values.
            &m,          // N: Order of the bidiagonal matrix.
            D_loc,       // D: Diagonal elements of B.
            E_loc,       // E: Subdiagonal elements of B.
            &zero,       // VL: Not referenced.
            &zero,       // VU: Not referenced.
            &li_zero,    // IL: Not referenced.
            &li_zero,    // IU: Not referenced.
            &ns,         // NS: No. of singular values found.
            s.data(),    // S: Array holding the singular values.
            nullptr,     // Z: Not referenced.
            &li_one,     // LDZ: Leading dimension of Z.
            workdbdsvd,  // WORK: Scratch space.
            iworkdbdsvd, // IWORK: Integer work containing indices of unconverged elements on failure.
            &info        // INFO: Exit code.
          );

          // Set the full matrix singular values
          s.resize(minmn);
          Sfull.setcol(s, i);

          //#pragma omp barrier
          double t1_svd = omp_get_wtime();
          if (tid == 0) {
            svd_time += t1_svd - t0_svd;
          }
        }
        // Free temporaries (if any)
        // BRD stage
        free(workbrd);

        // DBSVD stage
        free(workdbdsvd);
        free(iworkdbdsvd);
      }

      //std::cout << "Printing Dvecs" << std::endl;
      //for (size_t i = 0; i < minmn*nslices; i++) {
      //  if (i % minmn == 0) std::cout << std::endl;
      //  std::cout << Dvecs[i] << " ";
      //}
      //std::cout << std::endl;

      //std::cout << "Printing Evecs" << std::endl;
      //for (size_t i = 0; i < (minmn - 1) *nslices; i++) {
      //  if (i % (minmn - 1) == 0) std::cout << std::endl;
      //  std::cout << Evecs[i] << " ";
      //}
      //std::cout << std::endl;
    }
  }

  //std::cout << "Singular values computed." << std::endl;
  //Sfull.print();
  #pragma omp barrier
  t1 = omp_get_wtime();
  double stage1_time = t1 - t0;

  #pragma omp barrier
  t0 = omp_get_wtime();
  // Compute the thresholds
  std::vector<size_t> ks = thresholds(Sfull, tol);
  if (verbose) {
    std::cout << "Ranks found." << std::endl;
    for (size_t ii = 0; ii < ks.size(); ii++) {
      std::cout << ks[ii] << " ";
    }
    std::cout << std::endl;
  }
  Sfull.clear();
  #pragma omp barrier
  t1 = omp_get_wtime();
  double thr_time = t1 - t0;
  
  #pragma omp barrier
  t0 = omp_get_wtime();
  // STAGE 2: Recompute singular values and vectors with correct ranks

  // Create the output variables
  size_t U_fixed_dim = m;
  size_t U_buflen    = 0;
  for (size_t r = 0; r < ks.size(); r++) {
    U_buflen += U_fixed_dim * ks[r];
  }
  JaggedTensor U(U_buflen, U_fixed_dim, ks);

  size_t Vt_fixed_dim = n;
  size_t Vt_buflen    = 0;
  for (size_t r = 0; r < ks.size(); r++) {
    Vt_buflen += Vt_fixed_dim * ks[r];
  }
  JaggedTensor Vt(Vt_buflen, Vt_fixed_dim, ks, true);

  size_t S_buflen = std::accumulate(ks.begin(), ks.end(), (size_t) 0);
  JaggedMatrix S(S_buflen, ks);

  // Get dimensions and workspaces for paths
  lapack_int qrwork2 = 0, pbrwork = 0, qbrwork = 0;
  lapack_int dbdsvdwork2 = 0, dbdsvdiwork2 = 0, dbdsvdzwork = 0;

  // Find the largest rank
  size_t kmax        = *std::max_element(ks.begin(), ks.end());
  lapack_int li_kmax = kmax;

  if (m >= n) {
    if (m >= mnthr) { // Path 1: M >> N (approx M >= 1.6 N)
      // Optimal work size needed for bidiagonal SVD
      dbdsvdwork2  = 14*n;
      dbdsvdiwork2 = 12*n;
      dbdsvdzwork  = li_kmax*(n*2+1);

      // Workspace query for bidiagonal reduction stage (max work)
      lapack_int br_info, br_lwork;
      double br_wkopt;
      
      br_lwork = -1;
      dormbr("Q", "L", "N", &n, &li_kmax, &n, nullptr, &n, nullptr, 
        nullptr, &m, &br_wkopt, &br_lwork, &br_info);
      qbrwork = (lapack_int) br_wkopt;

      br_lwork = -1;
      dormbr("P", "R", "T", &li_kmax, &n, &n, nullptr, &n, nullptr, 
        nullptr, &li_kmax, &br_wkopt, &br_lwork, &br_info);
      pbrwork = (lapack_int) br_wkopt;

      // Workspace query for QR stage
      lapack_int qr_info, qr_lwork;
      double qr_wkopt;
      
      qr_lwork = -1;
      dormqr("L", "N", &m, &li_kmax, &n, nullptr, &m, nullptr, nullptr,
        &m, &qr_wkopt, &qr_lwork, &qr_info);
      qrwork2 = (lapack_int) qr_wkopt;
    } else { // Path 2 : M >= N
      // Optimal work size needed for bidiagonal SVD
      dbdsvdwork2  = 14*n;
      dbdsvdiwork2 = 12*n;
      dbdsvdzwork  = li_kmax*(n*2+1);

      // Workspace query for bidiagonal reduction stage (max work)
      lapack_int br_info, br_lwork;
      double br_wkopt;
      
      br_lwork = -1;
      dormbr("Q", "L", "N", &m, &li_kmax, &n, nullptr, &m, nullptr, 
        nullptr, &m, &br_wkopt, &br_lwork, &br_info);
      qbrwork = (lapack_int) br_wkopt;

      br_lwork = -1;
      dormbr("P", "R", "T", &li_kmax, &n, &m, nullptr, &m, nullptr, 
        nullptr, &li_kmax, &br_wkopt, &br_lwork, &br_info);
      pbrwork = (lapack_int) br_wkopt;
    }
  } else { 
    if (n >= mnthr) { // Path 1t: N >> M (approx N >= 1.6 M)
      // Optimal work size needed for bidiagonal SVD
      dbdsvdwork2  = 14*m;
      dbdsvdiwork2 = 12*m;
      dbdsvdzwork  = li_kmax*(m*2+1);

      // Workspace query for bidiagonal reduction stage (max work)
      lapack_int br_info, br_lwork;
      double br_wkopt;
      
      br_lwork = -1;
      dormbr("Q", "L", "N", &m, &li_kmax, &m, nullptr, &m, nullptr, 
        nullptr, &m, &br_wkopt, &br_lwork, &br_info);
      qbrwork = (lapack_int) br_wkopt;

      br_lwork = -1;
      dormbr("P", "R", "T", &li_kmax, &m, &m, nullptr, &m, nullptr, 
        nullptr, &li_kmax, &br_wkopt, &br_lwork, &br_info);
      pbrwork = (lapack_int) br_wkopt;

      // Workspace query for QR stage
      lapack_int qr_info, qr_lwork;
      double qr_wkopt;

      qr_lwork = -1;
      dormlq("R", "N", &li_kmax, &n, &m, nullptr, &m, nullptr, nullptr,
        &li_kmax, &qr_wkopt, &qr_lwork, &qr_info);
      qrwork2 = (lapack_int) qr_wkopt;
    } else { // Path 2t : N >= M
      // Optimal work size needed for bidiagonal SVD
      dbdsvdwork2  = 14*m;
      dbdsvdiwork2 = 12*m;
      dbdsvdzwork  = li_kmax*(m*2+1);

      // Workspace query for bidiagonal reduction stage (max work)
      lapack_int br_info, br_lwork;
      double br_wkopt;
      
      br_lwork = -1;
      dormbr("Q", "L", "N", &m, &li_kmax, &n, nullptr, &m, nullptr, 
        nullptr, &m, &br_wkopt, &br_lwork, &br_info);
      qbrwork = (lapack_int) br_wkopt;

      br_lwork = -1;
      dormbr("P", "R", "T", &li_kmax, &n, &m, nullptr, &m, nullptr, 
        nullptr, &li_kmax, &br_wkopt, &br_lwork, &br_info);
      pbrwork = (lapack_int) br_wkopt;
    }
  }

  if (verbose) {
    std::cout << "Stage 2: Workspace query outputs:" << std::endl;
    std::cout << "DBDSVD stage : " << dbdsvdwork2 << " " 
              << dbdsvdiwork2 << " " << dbdsvdzwork << std::endl;
    std::cout << "BRD stage    : " << qbrwork << " " << pbrwork << std::endl;
    std::cout << "QR stage     : " << qrwork2 << std::endl;
  }

  // Loop through the slices and compute the truncated SVD
  if (m >= n) {
    if (m >= mnthr) { // Path 1
      /*
        A = Q * R = Q * ( QB * B * PB**T )
                  = Q * ( QB * ( UB * S * VB**T ) * PB**T )
        U = Q * QB * UB; V**T = VB**T * PB**T
      */
      //std::cout << "Stage 2: Path 1" << std::endl;

#pragma omp parallel
      {
        // Create temporaries
        // DBDSVD stage
        double *workdbdsvd, *workdbdsvdz;
        lapack_int* iworkdbdsvd;
        workdbdsvd  = (double*) malloc(dbdsvdwork2 * sizeof(double));
        workdbdsvdz = (double*) malloc(dbdsvdzwork * sizeof(double));
        iworkdbdsvd = (lapack_int*) malloc(dbdsvdiwork2 * sizeof(lapack_int));

        // Computing U stage
        double *workqbr = (double*) malloc(qbrwork * sizeof(double));
        double *workqr2 = (double*) malloc(qrwork2 * sizeof(double));

        // Computing Vt stage
        double *workpbr = (double*) malloc(pbrwork * sizeof(double));

#pragma omp for
        for (size_t i = 0; i < nslices; i++) {
          if (ks[i] > 0) {
            // Inputs for DBDSVD
            //std::cout << "Printing Dvecs[" << i << "]" << std::endl;
            //for (size_t ii = 0; ii < minmn; ii++) {
            //  std::cout << Dvecs[i*minmn + ii] << " ";
            //}
            //std::cout << std::endl;

            //std::cout << "Printing Evecs[" << i << "]" << std::endl;
            //for (size_t ii = 0; ii < minmn-1; ii++) {
            //  std::cout << Evecs[i*(minmn-1) + ii] << " ";
            //}
            //std::cout << std::endl;
            
            // Outputs for this slice
            size_t k = ks[i];
            Matrix Uk(m*k, m, k);
            Matrix Vkt(k*n, k, n);
            std::vector<double> s(minmn+pad);
            
            lapack_int ldu = m, ldvt = k;

            // Ensure matrices are zero initialised
            std::memset(Uk.data_ptr, 0.0, m * k * sizeof(double));
            std::memset(Vkt.data_ptr, 0.0, k * n * sizeof(double));

            // DBDSVD stage (recalculate the singular values and vectors)
            double *D_loc = Dvecs + (i * minmn);
            double *E_loc = Evecs + (i * (minmn - 1));
            double zero   = 0.0;
            
            // LAPACK variables
            lapack_int info = 0;
            lapack_int ns = 0, il = 1, iu = k, ldz = 2*n;

            dbdsvdx(
              "U",         // UPLO: B is upper bidiagonal.
              "V",         // JOBZ: Compute singular values and vectors.
              "I",         // RANGE: Compute singular values in index range.
              &n,          // N: Order of the bidiagonal matrix.
              D_loc,       // D: Diagonal elements of B.
              E_loc,       // E: Superdiagonal elements of B.
              &zero,       // VL: Not referenced.
              &zero,       // VU: Not referenced.
              &il,         // IL: Lower index of singular values returned. 1 <= IL <= IU <= min(M, N).
              &iu,         // IU: Upper index of singular values returned. 1 <= IL <= IU <= min(M, N).
              &ns,         // NS: No. of singular values found.
              s.data(),    // S: Array holding the singular values.
              workdbdsvdz, // Z: Array containing the singular vectors.
              &ldz,        // LDZ: Leading dimension of Z.
              workdbdsvd,  // WORK: Scratch space.
              iworkdbdsvd, // IWORK: Integer work containing indices of unconverged elements on failure.
              &info        // INFO: Exit code.
            );

            // Resize the singular values to k
            s.resize(k);

            // Copy over UB and VB**T
            for (size_t jj = 0; jj < k; jj++) {
              // Go through the Z array column by column
              for (size_t ii = 0; ii < n; ii++) {
                Uk.set(ii, jj, workdbdsvdz[(jj * (2 * n)) + ii]);
                Vkt.set(jj, ii, workdbdsvdz[(jj * (2 * n)) + n + ii]);
              }
            }
          
            // Compute the left singular vectors : U = Q * QB * UB
            double *T_loc    = Tmats + (i * n * n);
            double *tauq_loc = tauq + (i * minmn);

            // LAPACK variables
            lapack_int nk = k;
            info = 0;
           
            // Compute QB * UB
            // Here C = UB
            dormbr(
              "Q",         // VECT: Applying a QB matrix to UB.
              "L",         // SIDE: From the left.
              "N",         // TRANS: Not transposed.
              &n,          // M: No. of rows of the matrix UB.
              &nk,         // N: No. of columns of the matrix UB.
              &n,          // K: No. of columns in matrix reduced by DGEBRD.
              T_loc,       // A: Matrix overwritten by DGEBRD.
              &n,          // LDA: Leading dimension of A.
              tauq_loc,    // TAU: Scalar factors of the QB reflector.
              Uk.data_ptr, // C: The UB matrix containing the left singular vectors.
              &ldu,        // LDC: Leading dimension of UB.
              workqbr,     // WORK: Scratch space.
              &qbrwork,    // LWORK: Dimension of the array WORK.
              &info        // INFO: Exit code.
            );

            // Compute Q * QB * UB
            double *A_loc   = A_copy.data_ptr + (i * slice_size);
            double *tau_loc = tau + (i * minmn);

            // LAPACK variables
            info = 0;

            dormqr(
              "L",         // SIDE: Apply a Q matrix from the left.
              "N",         // TRANS: Q is not transposed.
              &m,          // M: No. of rows QB * UB.
              &nk,         // N: No. of columns of QB * UB.
              &n,          // K: No. of elementary reflectors in Q.
              A_loc,       // A: Matrix overwritten by DGEQRF.
              &m,          // LDA: Leading dimension of A.
              tau_loc,     // TAU: Scalar factors of the Q reflector.
              Uk.data_ptr, // C: The QB * UB matrix containing the left singular vectors.
              &ldu,        // LDC: Leading dimension of C.
              workqr2,     // WORK: Scratch space.
              &qrwork2,    // LWORK: Dimension of the array WORK.
              &info        // INFO: Exit code. 
            );

            // Compute the right singular vectors: V**T = VB**T * PB**T
            double *taup_loc = taup + (i * minmn);

            // LAPACK variables
            nk = k;
            info = 0;
           
            // Compute VB**T * PB**T
            // Here C = VB**T
            dormbr(
              "P",          // VECT: Applying a PB**T matrix to VB**T.
              "R",          // SIDE: From the right.
              "T",          // TRANS: Transposed.
              &nk,          // M: No. of rows of the matrix VB**T.
              &n,           // N: No. of columns of the matrix VB**T.
              &n,           // K: No. of rows in matrix reduced by DGEBRD.
              T_loc,        // A: Matrix overwritten by DGEBRD.
              &n,           // LDA: Leading dimension of A.
              taup_loc,     // TAU: Scalar factors of the PB reflector.
              Vkt.data_ptr, // C: The UB matrix containing the left singular vectors.
              &ldvt,        // LDC: Leading dimension of UB.
              workpbr,      // WORK: Scratch space.
              &pbrwork,     // LWORK: Dimension of the array WORK.
              &info         // INFO: Exit code.
            );

            // Save the output
            U.setfrontalslice(Uk, i);
            S.setcol(s, i);
            Vt.setfrontalslice(Vkt, i);

            // Clear the temporaries
            Uk.clear();
            Vkt.clear();
          }
        }
        // Free temporaries (if any)
        // DBDSVD stage
        free(workdbdsvdz);
        free(workdbdsvd);
        free(iworkdbdsvd);

        // Computing U
        free(workqbr);
        free(workqr2);

        // Computing Vt
        free(workpbr);
      }

      // Free saved computations
      // QR stage
      free(tau);

      // BRD stage
      free(Tmats);
      free(tauq);
      free(taup);
      free(Dvecs);
      free(Evecs);

    } else { // Path 2
      /*
        A = QB * B * PB**T = QB * ( UB * S * VB**T ) * PB**T
        U = QB * UB; V**T = VB**T * PB**T
      */
      //std::cout << "Stage 2: Path 2" << std::endl;

#pragma omp parallel
      {
        // Create temporaries
        double *workdbdsvd, *workdbdsvdz;
        lapack_int* iworkdbdsvd;
        workdbdsvd  = (double*) malloc(dbdsvdwork2 * sizeof(double));
        workdbdsvdz = (double*) malloc(dbdsvdzwork * sizeof(double));
        iworkdbdsvd = (lapack_int*) malloc(dbdsvdiwork2 * sizeof(lapack_int));

        // Computing U stage
        double *workqbr = (double*) malloc(qbrwork * sizeof(double));

        // Computing Vt stage
        double *workpbr = (double*) malloc(pbrwork * sizeof(double));

#pragma omp for
        for (size_t i = 0; i < nslices; i++) {
          if (ks[i] > 0) {
            // Inputs for DBDSVD
            //std::cout << "Printing Dvecs[" << i << "]" << std::endl;
            //for (size_t ii = 0; ii < minmn; ii++) {
            //  std::cout << Dvecs[i*minmn + ii] << " ";
            //}
            //std::cout << std::endl;

            //std::cout << "Printing Evecs[" << i << "]" << std::endl;
            //for (size_t ii = 0; ii < minmn-1; ii++) {
            //  std::cout << Evecs[i*(minmn-1) + ii] << " ";
            //}
            //std::cout << std::endl;
            
            // Outputs for this slice
            size_t k = ks[i];
            Matrix Uk(m*k, m, k);
            Matrix Vkt(k*n, k, n);
            std::vector<double> s(minmn+pad);
            
            lapack_int ldu = m, ldvt = k;

            // Ensure matrices are zero initialised
            std::memset(Uk.data_ptr, 0.0, m * k * sizeof(double));
            std::memset(Vkt.data_ptr, 0.0, k * n * sizeof(double));

            // DBDSVD stage (recalculate the singular values and vectors)
            double *D_loc = Dvecs + (i * minmn);
            double *E_loc = Evecs + (i * (minmn - 1));
            double zero   = 0.0;
            
            // LAPACK variables
            lapack_int info = 0;
            lapack_int ns = 0, il = 1, iu = k, ldz = 2*n;

            dbdsvdx(
              "U",         // UPLO: B is upper bidiagonal.
              "V",         // JOBZ: Compute singular values and vectors.
              "I",         // RANGE: Compute singular values in index range.
              &n,          // N: Order of the bidiagonal matrix.
              D_loc,       // D: Diagonal elements of B.
              E_loc,       // E: Superdiagonal elements of B.
              &zero,       // VL: Not referenced.
              &zero,       // VU: Not referenced.
              &il,         // IL: Lower index of singular values returned. 1 <= IL <= IU <= min(M, N).
              &iu,         // IU: Upper index of singular values returned. 1 <= IL <= IU <= min(M, N).
              &ns,         // NS: No. of singular values found.
              s.data(),    // S: Array holding the singular values.
              workdbdsvdz, // Z: Array containing the singular vectors.
              &ldz,        // LDZ: Leading dimension of Z.
              workdbdsvd,  // WORK: Scratch space.
              iworkdbdsvd, // IWORK: Integer work containing indices of unconverged elements on failure.
              &info        // INFO: Exit code.
            );

            // Resize the singular values to k
            s.resize(k);

            // Copy over UB and VB**T
            for (size_t jj = 0; jj < k; jj++) {
              // Go through the Z array column by column
              for (size_t ii = 0; ii < n; ii++) {
                Uk.set(ii, jj, workdbdsvdz[(jj * (2 * n)) + ii]);
                Vkt.set(jj, ii, workdbdsvdz[(jj * (2 * n)) + n + ii]);
              }
            }
            
            // Compute the left singular vectors : U = Q * QB * UB
            double *A_loc   = A_copy.data_ptr + (i * slice_size);
            double *tauq_loc = tauq + (i * minmn);

            // LAPACK variables
            lapack_int nk = k;
            info = 0;
           
            // Compute QB * UB
            // Here C = UB
            dormbr(
              "Q",         // VECT: Applying a QB matrix to UB.
              "L",         // SIDE: From the left.
              "N",         // TRANS: Not transposed.
              &m,          // M: No. of rows of the matrix UB.
              &nk,         // N: No. of columns of the matrix UB.
              &n,          // K: No. of columns in matrix reduced by DGEBRD.
              A_loc,       // A: Matrix overwritten by DGEBRD.
              &m,          // LDA: Leading dimension of A.
              tauq_loc,    // TAU: Scalar factors of the QB reflector.
              Uk.data_ptr, // C: The UB matrix containing the left singular vectors.
              &ldu,        // LDC: Leading dimension of UB.
              workqbr,     // WORK: Scratch space.
              &qbrwork,    // LWORK: Dimension of the array WORK.
              &info        // INFO: Exit code.
            );

            // Compute the right singular vectors: V**T = VB**T * PB**T
            double *taup_loc = taup + (i * minmn);

            // LAPACK variables
            nk = k;
            info = 0;
           
            // Compute VB**T * PB**T
            // Here C = VB**T
            dormbr(
              "P",          // VECT: Applying a PB**T matrix to VB**T.
              "R",          // SIDE: From the right.
              "T",          // TRANS: Transposed.
              &nk,          // M: No. of rows of the matrix VB**T.
              &n,           // N: No. of columns of the matrix VB**T.
              &m,           // K: No. of rows in matrix reduced by DGEBRD.
              A_loc,        // A: Matrix overwritten by DGEBRD.
              &m,           // LDA: Leading dimension of A.
              taup_loc,     // TAU: Scalar factors of the PB reflector.
              Vkt.data_ptr, // C: The UB matrix containing the left singular vectors.
              &ldvt,        // LDC: Leading dimension of UB.
              workpbr,      // WORK: Scratch space.
              &pbrwork,     // LWORK: Dimension of the array WORK.
              &info         // INFO: Exit code.
            );

            // Save the output
            U.setfrontalslice(Uk, i);
            S.setcol(s, i);
            Vt.setfrontalslice(Vkt, i);

            // Clear the temporaries
            Uk.clear();
            Vkt.clear();
          }
        }
        // Free temporaries (if any)
        // DBDSVD stage
        free(workdbdsvdz);
        free(workdbdsvd);
        free(iworkdbdsvd);

        // Computing U
        free(workqbr);

        // Computing Vt
        free(workpbr);
      }
      // Free saved computations
      // BRD stage
      free(tauq);
      free(taup);
      free(Dvecs);
      free(Evecs);
    }
  } else {
    if (n >= mnthr) { // Path 1t
      /*
        A = L * Q = ( QB * B * PB**T ) * Q
                  = ( QB * ( UB * S * VB**T ) * PB**T ) * Q
        U = QB * UB ; V**T = VB**T * PB**T * Q
      */
      //std::cout << "Stage 2: Path 1t" << std::endl;

#pragma omp parallel
      {
        // Create temporaries
        // DBDSVD stage
        double *workdbdsvd, *workdbdsvdz;
        lapack_int* iworkdbdsvd;
        workdbdsvd  = (double*) malloc(dbdsvdwork2 * sizeof(double));
        workdbdsvdz = (double*) malloc(dbdsvdzwork * sizeof(double));
        iworkdbdsvd = (lapack_int*) malloc(dbdsvdiwork2 * sizeof(lapack_int));

        // Computing U stage
        double *workqbr = (double*) malloc(qbrwork * sizeof(double));

        // Computing Vt stage
        double *workpbr = (double*) malloc(pbrwork * sizeof(double));
        double *workqr2 = (double*) malloc(qrwork2 * sizeof(double));

#pragma omp for
        for (size_t i = 0; i < nslices; i++) {
          if (ks[i] > 0) {
            // Inputs for DBDSVD
            //std::cout << "Printing Dvecs[" << i << "]" << std::endl;
            //for (size_t ii = 0; ii < minmn; ii++) {
            //  std::cout << Dvecs[i*minmn + ii] << " ";
            //}
            //std::cout << std::endl;

            //std::cout << "Printing Evecs[" << i << "]" << std::endl;
            //for (size_t ii = 0; ii < minmn-1; ii++) {
            //  std::cout << Evecs[i*(minmn-1) + ii] << " ";
            //}
            //std::cout << std::endl;
            
            // Outputs for this slice
            size_t k = ks[i];
            Matrix Uk(m*k, m, k);
            Matrix Vkt(k*n, k, n);
            std::vector<double> s(minmn+pad);
            
            lapack_int ldu = m, ldvt = k;

            // Ensure matrices are zero initialised
            std::memset(Uk.data_ptr, 0.0, m * k * sizeof(double));
            std::memset(Vkt.data_ptr, 0.0, k * n * sizeof(double));

            // DBDSVD stage (recalculate the singular values and vectors)
            double *D_loc = Dvecs + (i * minmn);
            double *E_loc = Evecs + (i * (minmn - 1));
            double zero   = 0.0;
            
            // LAPACK variables
            lapack_int info = 0;
            lapack_int ns = 0, il = 1, iu = k, ldz = 2*m;

            dbdsvdx(
              "U",         // UPLO: B is upper bidiagonal.
              "V",         // JOBZ: Compute singular values and vectors.
              "I",         // RANGE: Compute singular values in index range.
              &m,          // N: Order of the bidiagonal matrix.
              D_loc,       // D: Diagonal elements of B.
              E_loc,       // E: Superdiagonal elements of B.
              &zero,       // VL: Not referenced.
              &zero,       // VU: Not referenced.
              &il,         // IL: Lower index of singular values returned. 1 <= IL <= IU <= min(M, N).
              &iu,         // IU: Upper index of singular values returned. 1 <= IL <= IU <= min(M, N).
              &ns,         // NS: No. of singular values found.
              s.data(),    // S: Array holding the singular values.
              workdbdsvdz, // Z: Array containing the singular vectors.
              &ldz,        // LDZ: Leading dimension of Z.
              workdbdsvd,  // WORK: Scratch space.
              iworkdbdsvd, // IWORK: Integer work containing indices of unconverged elements on failure.
              &info        // INFO: Exit code.
            );

            // Resize the singular values to k
            s.resize(k);

            // Copy over UB and VB**T
            for (size_t jj = 0; jj < k; jj++) {
              // Go through the Z array column by column
              for (size_t ii = 0; ii < m; ii++) {
                Uk.set(ii, jj, workdbdsvdz[(jj * (2 * m)) + ii]);
                Vkt.set(jj, ii, workdbdsvdz[(jj * (2 * m)) + m + ii]);
              }
            }
          
            // Compute the left singular vectors : U = QB * UB
            double *T_loc    = Tmats + (i * m * m);
            double *tauq_loc = tauq + (i * minmn);

            // LAPACK variables
            lapack_int nk = k;
            info = 0;
           
            // Compute QB * UB
            // Here C = UB
            dormbr(
              "Q",         // VECT: Applying a QB matrix to UB.
              "L",         // SIDE: From the left.
              "N",         // TRANS: Not transposed.
              &m,          // M: No. of rows of the matrix UB.
              &nk,         // N: No. of columns of the matrix UB.
              &m,          // K: No. of columns in matrix reduced by DGEBRD.
              T_loc,       // A: Matrix overwritten by DGEBRD.
              &m,          // LDA: Leading dimension of A.
              tauq_loc,    // TAU: Scalar factors of the QB reflector.
              Uk.data_ptr, // C: The UB matrix containing the left singular vectors.
              &ldu,        // LDC: Leading dimension of UB.
              workqbr,     // WORK: Scratch space.
              &qbrwork,    // LWORK: Dimension of the array WORK.
              &info        // INFO: Exit code.
            );

            // Compute the right singular vectors: V**T = VB**T * PB**T * Q
            double *taup_loc = taup + (i * minmn);

            // LAPACK variables
            nk = k;
            info = 0;
           
            // Compute VB**T * PB**T
            // Here C = VB**T
            dormbr(
              "P",          // VECT: Applying a PB**T matrix to VB**T.
              "R",          // SIDE: From the right.
              "T",          // TRANS: Transposed.
              &nk,          // M: No. of rows of the matrix VB**T.
              &m,           // N: No. of columns of the matrix VB**T.
              &m,           // K: No. of rows in matrix reduced by DGEBRD.
              T_loc,        // A: Matrix overwritten by DGEBRD.
              &m,           // LDA: Leading dimension of A.
              taup_loc,     // TAU: Scalar factors of the PB reflector.
              Vkt.data_ptr, // C: The VB**T matrix containing the right singular vectors.
              &ldvt,        // LDC: Leading dimension of C.
              workpbr,      // WORK: Scratch space.
              &pbrwork,     // LWORK: Dimension of the array WORK.
              &info         // INFO: Exit code.
            );

            // Compute VB**T * PB**T * Q
            double *A_loc   = A_copy.data_ptr + (i * slice_size);
            double *tau_loc = tau + (i * minmn);

            // LAPACK variables
            info = 0;

            // Vkt is k x n and is C in this context
            // C = [VB**T * PB**T 0]
            dormlq(
              "R",          // SIDE: Apply a Q matrix to the right.
              "N",          // TRANS: Q is not transposed.
              &nk,          // M: No. of rows of the matrix C.
              &n,           // N: No. of columns of the matrix C.
              &m,           // K: No. of elementary reflectors in Q.
              A_loc,        // A: Matrix overwritten by DGELQF.
              &m,           // LDA: Leading dimension of A.
              tau_loc,      // TAU: Scalar factors of the Q reflector.
              Vkt.data_ptr, // C: The Vkt matrix containing the right singular vectors.
              &ldvt,        // LDC: Leading dimension of C.
              workqr2,      // WORK: Scratch space.
              &qrwork2,     // LWORK: Dimension of the array WORK.
              &info         // INFO: Exit code.
            );

            // Save the output
            U.setfrontalslice(Uk, i);
            S.setcol(s, i);
            Vt.setfrontalslice(Vkt, i);

            // Clear the temporaries
            Uk.clear();
            Vkt.clear();
          }
        }
        // Free temporaries (if any)
        // DBDSVD stage
        free(workdbdsvdz);
        free(workdbdsvd);
        free(iworkdbdsvd);

        // Computing U
        free(workqbr);

        // Computing Vt
        free(workpbr);
        free(workqr2);
      }
      // Free saved computations
      // QR stage
      free(tau);

      // BRD stage
      free(Tmats);
      free(tauq);
      free(taup);
      free(Dvecs);
      free(Evecs);
    } else { // Path 2t
      /*
        A = QB * B * PB**T = QB * ( UB * S * VB**T ) * PB**T
        U = QB * UB; V**T = VB**T * PB**T
      */
      //std::cout << "Stage 2: Path 2t" << std::endl;

#pragma omp parallel
      {
        // Create temporaries
        // DBDSVD stage
        double *workdbdsvd, *workdbdsvdz;
        lapack_int* iworkdbdsvd;
        workdbdsvd  = (double*) malloc(dbdsvdwork2 * sizeof(double));
        workdbdsvdz = (double*) malloc(dbdsvdzwork * sizeof(double));
        iworkdbdsvd = (lapack_int*) malloc(dbdsvdiwork2 * sizeof(lapack_int));

        // Computing U stage
        double *workqbr = (double*) malloc(qbrwork * sizeof(double));

        // Computing Vt stage
        double *workpbr = (double*) malloc(pbrwork * sizeof(double));

#pragma omp for
        for (size_t i = 0; i < nslices; i++) {
          if (ks[i] > 0) {
            // Inputs for DBDSVD
            //std::cout << "Printing Dvecs[" << i << "]" << std::endl;
            //for (size_t ii = 0; ii < minmn; ii++) {
            //  std::cout << Dvecs[i*minmn + ii] << " ";
            //}
            //std::cout << std::endl;

            //std::cout << "Printing Evecs[" << i << "]" << std::endl;
            //for (size_t ii = 0; ii < minmn-1; ii++) {
            //  std::cout << Evecs[i*(minmn-1) + ii] << " ";
            //}
            //std::cout << std::endl;
            
            // Outputs for this slice
            size_t k = ks[i];
            Matrix Uk(m*k, m, k);
            Matrix Vkt(k*n, k, n);
            std::vector<double> s(minmn+pad);
            
            lapack_int ldu = m, ldvt = k;

            // Ensure matrices are zero initialised
            std::memset(Uk.data_ptr, 0.0, m * k * sizeof(double));
            std::memset(Vkt.data_ptr, 0.0, k * n * sizeof(double));

            // DBDSVD stage (recalculate the singular values and vectors)
            double *D_loc = Dvecs + (i * minmn);
            double *E_loc = Evecs + (i * (minmn - 1));
            double zero   = 0.0;
            
            // LAPACK variables
            lapack_int info = 0;
            lapack_int ns = 0, il = 1, iu = k, ldz = 2*m;

            dbdsvdx(
              "L",         // UPLO: B is lower bidiagonal.
              "V",         // JOBZ: Compute singular values and vectors.
              "I",         // RANGE: Compute singular values in index range.
              &m,          // N: Order of the bidiagonal matrix.
              D_loc,       // D: Diagonal elements of B.
              E_loc,       // E: Subdiagonal elements of B.
              &zero,       // VL: Not referenced.
              &zero,       // VU: Not referenced.
              &il,         // IL: Lower index of singular values returned. 1 <= IL <= IU <= min(M, N).
              &iu,         // IU: Upper index of singular values returned. 1 <= IL <= IU <= min(M, N).
              &ns,         // NS: No. of singular values found.
              s.data(),    // S: Array holding the singular values.
              workdbdsvdz, // Z: Array containing the singular vectors.
              &ldz,        // LDZ: Leading dimension of Z.
              workdbdsvd,  // WORK: Scratch space.
              iworkdbdsvd, // IWORK: Integer work containing indices of unconverged elements on failure.
              &info        // INFO: Exit code.
            );

            // Resize the singular values to k
            s.resize(k);

            // Copy over UB and VB**T
            for (size_t jj = 0; jj < k; jj++) {
              // Go through the Z array column by column
              for (size_t ii = 0; ii < m; ii++) {
                Uk.set(ii, jj, workdbdsvdz[(jj * (2 * m)) + ii]);
                Vkt.set(jj, ii, workdbdsvdz[(jj * (2 * m)) + m + ii]);
              }
            }
          
            // Compute the left singular vectors : U = QB * UB
            double *A_loc    = A_copy.data_ptr + (i * slice_size);
            double *tauq_loc = tauq + (i * minmn);

            // LAPACK variables
            lapack_int nk = k;
            info = 0;
           
            // Compute QB * UB
            // Here C = UB
            dormbr(
              "Q",         // VECT: Applying a QB matrix to UB.
              "L",         // SIDE: From the left.
              "N",         // TRANS: Not transposed.
              &m,          // M: No. of rows of the matrix UB.
              &nk,         // N: No. of columns of the matrix UB.
              &n,          // K: No. of columns in matrix reduced by DGEBRD.
              A_loc,       // A: Matrix overwritten by DGEBRD.
              &m,          // LDA: Leading dimension of A.
              tauq_loc,    // TAU: Scalar factors of the QB reflector.
              Uk.data_ptr, // C: The UB matrix containing the left singular vectors.
              &ldu,        // LDC: Leading dimension of UB.
              workqbr,     // WORK: Scratch space.
              &qbrwork,    // LWORK: Dimension of the array WORK.
              &info        // INFO: Exit code.
            );

            // Compute the right singular vectors: V**T = VB**T * PB**T * Q
            double *taup_loc = taup + (i * minmn);

            // LAPACK variables
            nk = k;
            info = 0;
           
            // Compute VB**T * PB**T
            // Here C = VB**T
            dormbr(
              "P",          // VECT: Applying a PB**T matrix to VB**T.
              "R",          // SIDE: From the right.
              "T",          // TRANS: Transposed.
              &nk,          // M: No. of rows of the matrix VB**T.
              &n,           // N: No. of columns of the matrix VB**T.
              &m,           // K: No. of rows in matrix reduced by DGEBRD.
              A_loc,        // A: Matrix overwritten by DGEBRD.
              &m,           // LDA: Leading dimension of A.
              taup_loc,     // TAU: Scalar factors of the PB reflector.
              Vkt.data_ptr, // C: The VB**T matrix containing the right singular vectors.
              &ldvt,        // LDC: Leading dimension of C.
              workpbr,      // WORK: Scratch space.
              &pbrwork,     // LWORK: Dimension of the array WORK.
              &info         // INFO: Exit code.
            );

            // Save the output
            U.setfrontalslice(Uk, i);
            S.setcol(s, i);
            Vt.setfrontalslice(Vkt, i);

            // Clear the temporaries
            Uk.clear();
            Vkt.clear();
          }
        }
        // Free temporaries (if any)
        // DBDSVD stage
        free(workdbdsvdz);
        free(workdbdsvd);
        free(iworkdbdsvd);

        // Computing U
        free(workqbr);

        // Computing Vt
        free(workpbr);
      }
      // Free saved computations
      // BRD stage
      free(tauq);
      free(taup);
      free(Dvecs);
      free(Evecs);
    }
  }
  #pragma omp barrier
  t1 = omp_get_wtime();
  double stage2_time = t1 - t0;

  #pragma omp barrier
  t0 = omp_get_wtime();

  // Undo scaling
  if (lscl) {
    lapack_int li_sbuflen = S_buflen;
    if (anrm > 0.0 && anrm < smlnum) {
      //std::cout << "Reached SMLNUM" << std::endl;
      dlascl(
        "G",         // G: A is a full matrix.
        &li_zero,    // KL: Not referenced.
        &li_zero,    // KU: Not referenced.
        &smlnum,     // CFROM: Matrix scaled as A(i, j) * CTO/CFROM.
        &anrm,       // CTO: Matrix scaled as A(i, j) * CTO/CFROM.
        &li_sbuflen, // M: No. of rows of the matrix. Treat as a long array.
        &li_one,     // N: No. of columns of the matrix. Treat as a long array.
        S.data_ptr,  // A: The data array.
        &li_sbuflen, // LDA: Leading dimension of A.
        &info        // INFO: Exit code.
      );
    } else if (anrm > bignum) {
      //std::cout << "Reached BIGNUM" << std::endl;
      dlascl( 
        "G",         // G: A is a full matrix.
        &li_zero,    // KL: Not referenced.
        &li_zero,    // KU: Not referenced.
        &bignum,     // CFROM: Matrix scaled as A(i, j) * CTO/CFROM.
        &anrm,       // CTO: Matrix scaled as A(i, j) * CTO/CFROM.
        &li_sbuflen, // M: No. of rows of the matrix. Treat as a long array.    ,
        &li_one,     // N: No. of columns of the matrix. Treat as a long array.
        S.data_ptr,  // A: The data array.
        &li_sbuflen, // LDA: Leading dimension of A.
        &info        // INFO: Exit code.
      );
    }
  }

  // Free the copied tensor
  A_copy.clear();

  #pragma omp barrier
  t1 = omp_get_wtime();
  double post_time = t1 - t0;

  std::cout << "Pre time    : " << pre_time << std::endl
            << "Stage 1 time: " << stage1_time << std::endl
            << "  QR time   : " << qr_time << std::endl
            << "  BRD time  : " << brd_time << std::endl
            << "  SVD time  : " << svd_time << std::endl
            << "Thr time    : " << thr_time << std::endl
            << "Stage 2 time: " << stage2_time << std::endl
            << "Post time   : " << post_time << std::endl;

  return std::make_tuple(std::move(U), std::move(S), std::move(Vt));
}

std::tuple<JaggedTensor, JaggedMatrix, JaggedTensor> tsvdmii_compress(const Tensor &A, 
        std::vector<Matrix> M, double tol) {
    std::vector<int> order;
    for(int i = 0; i < A.ndim; i++){
        if(i < 2) continue;
        else order.push_back(i);
    }
    Tensor A_hat = transform(A, M, order);

    // Compute the slicewise ranks
    Matrix Sv    = slicewise_svdvals(A_hat);
    std::vector<size_t> ks = thresholds(Sv, tol);

    // Perform the SVD
    JaggedTensor U, Vt;
    JaggedMatrix S;
    std::tie(U, S, Vt) = slicewise_svdks(A_hat, ks);

    // Clear temporary stuff
    A_hat.clear();
    Sv.clear();

    return std::make_tuple(std::move(U), std::move(S), std::move(Vt));
}

std::tuple<Tensor, Matrix, Tensor> tsvdmi_compress(const Tensor &A, std::vector<Matrix> M, int k) {
    std::vector<int> order;
    for(int i = 0; i < A.ndim; i++){
        if(i < 2) continue;
        else order.push_back(i);
    }
    Tensor A_hat = transform(A, M, order);

    Tensor U, Vt;
    Matrix S;
    std::tie(U, S, Vt) = slicewise_svdx(A_hat, k);
    A_hat.clear();
    return std::make_tuple(std::move(U), std::move(S), std::move(Vt));
}

Tensor tsvdmi_reconstruct(const Tensor& U, const Matrix& S, const Tensor& VT, std::vector<Matrix> M){
    Tensor A_hat = slicewise_matmul(U, S, VT);   
    std::vector<int> order;
    for(int i = 0; i < A_hat.ndim; i++){
        if(i < 2) continue;
        else order.push_back(i);
    }
    Tensor A_tilde = transform(A_hat, M, order);
    A_hat.clear();
    return A_tilde;
}

Matrix tensor_contract_all_but_one(Tensor& A, Tensor& B, size_t mode, bool naive=false){
    /*
    *
    * Unfolds two tensors A and B in the same mode, and computes A_(k) B_(k)^T, storing the result C
    * in column-major order.
    *   - Instead of manually unfolding these two tensors, we can get "blocks" from each of the buffers, multiply them, 
    *     and then sum the result.
    *   - In CblasColMajor mode, treat A's memory as A^T (num_cols x n_k, lda = n_k)
    *     and B's memory as B (p x n_k, ldb = n).
    *   - We want C (n_k x p, col-major, ldc = m) = A * B^T = (A^T)^T * (B)^T
    *   - So call: C(n_k x p) = Trans(A^T) * Trans(B)
    *     i.e., cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans, n_k, , k,
    *                       1.0, A, k, B, n, 0.0, C, m)
    */
    
    size_t i = 0; 

    // Take into account the mode which both are unfolding for. Remember this is A(k) (B(k))^T
    // The dimensions of the final tensor will be n_k x p. 
    // Get necessary dimensions
    std::vector<size_t> A_dims = A.getdims();
    std::vector<size_t> B_dims = B.getdims();

    size_t n_k = A_dims[mode];
    size_t p = B_dims[mode];

    // Determine the number of columns for each block and set dimensions accordingly. 
    size_t num_cols = 1;
    for (i = 0; i < mode; ++i) { 
      assert(A_dims[i] == B_dims[i] && "Dimensions of A and B must be equal except at n_k.");
      num_cols *= A_dims[i];
    }

    // This will be a cumulative matrix, makes the buffer all zeros to start.
    Matrix C(p*n_k, n_k, p);
    std::memset(C.data_ptr, 0, C.buflen * sizeof(double));

    // Get the number of "blocks" we are multiplying.
    size_t num_blocks = 1;
    for (i = mode + 1; i < A.ndim; ++i){ 
      assert(A_dims[i] == B_dims[i] && "Dimensions of A and B must be equal except at n_k.");
      num_blocks *= A_dims[i];
    }

    // Matrix multiply each block and sum in C.
    // A blocks have dimensions n_k X num_cols, B blocks have dimensions p X num_cols, hence why we transpose B by default.
    // std::cout << "p = " << p << "\nn_k = " << n_k << "\nnum_blocks = " << num_blocks << "\nnum_cols = " << num_cols << std::endl;

    if(naive){
      for (i = 0; i < num_blocks; ++i) {
        cblas_dgemm(
            CblasColMajor,                
            CblasTrans,                      
            CblasNoTrans,                         
            n_k,                                  
            p,                                
            num_cols,                          
            1.0,
            A.data_ptr + i * (n_k * num_cols),  // Change the start of the data pointer based on which block we are on.
            num_cols,                           // lda, since it's row-major, it's the number of columns.
            B.data_ptr + i * (p   * num_cols),
            num_cols,                            // ldb, since it's row-major, it's the number of columns.
            1.0,                                // accumulate into C
            C.data_ptr,
            n_k                              // ldc,  columns of C.
        );
      }
    } else {

      // Pin MKL to 1 thread per outer-OpenMP thread to avoid nested libgomp/mkl_gnu_thread
      // oversubscription.
      mkl_set_num_threads_local(1);

      // parallel version.
      double* C_data_ptr = C.data_ptr;
      size_t C_buflen = C.buflen;

      #pragma omp parallel for reduction(+:C_data_ptr[:C_buflen])
      for (size_t b = 0; b < num_blocks; ++b) {
        cblas_dgemm(
            CblasColMajor,                
            CblasTrans,                      
            CblasNoTrans,                         // B_block transposed
            n_k,                                  // m = rows of C
            p,                                // n = cols of C
            num_cols,                           // k = inner dim
            1.0,
            A.data_ptr + b * (n_k * num_cols),  // Change the start of the data pointer based on which block we are on.
            num_cols,                           // lda, since it's row-major, it's the number of columns.
            B.data_ptr + b * (p   * num_cols),
            num_cols,                            // ldb, since it's row-major, it's the number of columns.
            1.0,                                // accumulate into C
            C_data_ptr,
            n_k                              // ldc,  columns of C.
        );
      }
    }
    
    mkl_set_num_threads_local(0);

    return C;
}

Matrix tensor_contract_all_but_one_batched(const Tensor& A, const Tensor& B, size_t mode, size_t num_threads=3){
    /*
    * SAME AS TENSOR_CONTRACT_ALL_BUT_ONE, but instead, it's batched.
    * Unfolds two tensors A and B in the same mode, and computes A_(k) B_(k)^T, storing the result C
    * in column-major order.
    *   - Instead of manually unfolding these two tensors, we can get "blocks" from each of the buffers, multiply them, 
    *     and then sum the result.
    *   - In CblasColMajor mode, treat A's memory as A^T (num_cols x n_k, lda = n_k)
    *     and B's memory as B (p x n_k, ldb = n).
    *   - We want C (n_k x p, col-major, ldc = m) = A * B^T = (A^T)^T * (B)^T
    *   - So call: C(n_k x p) = Trans(A^T) * Trans(B)
    *     i.e., cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans, n_k, , k,
    *                       1.0, A, k, B, n, 0.0, C, m)
    */
    
    size_t i = 0; 
    // Just making a temporary variable to store the number of "threads" we are using. 

    // Take into account the mode which both are unfolding for. Remember this is A(k) (B(k))^T
    // The dimensions of the final tensor will be n_k x p. 
    // Get necessary dimensions
    std::vector<size_t> A_dims = A.dims;
    std::vector<size_t> B_dims = B.dims;

    size_t n_k = A_dims[mode];
    size_t p = B_dims[mode];

    // Determine the number of columns for each block and set dimensions accordingly. 
    size_t num_cols = 1;
    for (i = 0; i < mode; ++i) { 
      assert(A_dims[i] == B_dims[i] && "Dimensions of A and B must be equal except at n_k.");
      num_cols *= A_dims[i];
    }

    // Get the number of "blocks" we are multiplying.
    size_t num_blocks = 1;
    for (i = mode + 1; i < A.ndim; ++i){ 
      assert(A_dims[i] == B_dims[i] && "Dimensions of A and B must be equal except at n_k.");
      num_blocks *= A_dims[i];
    }

    // The number of "leftover blocks", meaning the number of blocks in the last iteration which are not covered by all threads.
    // I.e. if num_blocks = 26 and num_threads = 4, then only the first two threads will have blocks to multiply on the last iteration.
    
    size_t leftover_blocks = num_blocks % num_threads;
    // Edge case for very small tensors.
    if (leftover_blocks == num_blocks){
      num_threads = num_blocks;
      leftover_blocks = 0;
    }

    // This will be a cumulative matrix, makes the buffer all zeros to start.
    // Storing as one matrix with n_k rows and p*num_threads columns, because there are num_threads "blocks", which will be summed later.
    Matrix C(p*n_k*num_threads, n_k, p*num_threads);
    std::memset(C.data_ptr, 0, C.buflen * sizeof(double));

    // MKL Strided Batched BLAS documentation: 
    // https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/cblas-gemm-batch-strided.html
    // A blocks have dimensions n_k X num_cols, B blocks have dimensions p X num_cols, hence why we transpose B by default.
    MKL_INT cblas_m = (MKL_INT) n_k;
    MKL_INT cblas_k = (MKL_INT) num_cols;
    MKL_INT cblas_n = (MKL_INT) p;
    double cblas_alpha = 1.0;
    double cblas_beta = 1.0;
    double* cblas_a = A.data_ptr;
    MKL_INT cblas_lda = (MKL_INT) num_cols; 
    MKL_INT cblas_stridea = (MKL_INT) (n_k * num_cols);
    double* cblas_b = B.data_ptr;
    MKL_INT cblas_ldb = (MKL_INT) num_cols;
    MKL_INT cblas_strideb = (MKL_INT) (p * num_cols);
    double* cblas_c = C.data_ptr; 
    MKL_INT cblas_ldc = (MKL_INT) n_k;
    MKL_INT cblas_stridec = (MKL_INT) n_k * p;
    MKL_INT cblas_batch_size = (MKL_INT) num_threads;


    // TODO: CONSIDER CASES WHERE num_blocks % num_threads != 0.
    for(i = 0; i < num_blocks; i += num_threads){

      // Modify the batch size for the last set of blocks where not all threads have an assigned block.
      size_t batch_size = std::min(num_threads, num_blocks - i);
      cblas_batch_size = (MKL_INT) batch_size;
      // Matrix multiply each block and sum in C.
      cblas_dgemm_batch_strided(
          CblasColMajor, // Column major order. `Layout` parameter of MKL cblas call.
          CblasTrans, // A matrix is not transpose. `transa` param of MKL cblas call.
          CblasNoTrans, // B matrix is transpose. `transb` param of MKL cblas call.
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
      // Update beginning of buffer pointers. 
      cblas_a += (n_k * num_cols * num_threads);
      cblas_b += (p * num_cols * num_threads);
    }

    // TODO: SUM C MATRICES IN PARALLEL.
    // Non-parallel implementation just to test the original, for now.
    Matrix C_sum(p*n_k, n_k, p);
    std::memset(C_sum.data_ptr, 0, C_sum.buflen * sizeof(double));

    // #pragma omp parallel for
    for(i = 0; i < C_sum.buflen; ++i){
      size_t offset;
      for(size_t t = 0; t < num_threads; ++t){
        // find where in C this index is. 
        offset = i + (t * p*n_k);
        C_sum.data_ptr[i] += C.data_ptr[offset];
        // std::cout << "C.data_ptr[" << offset << "] = " << C.data_ptr[offset] << std::endl;

      }
    }
    // C is no longer needed. 
    C.clear();
    return C_sum;
}


Matrix tensor_contract_all_but_one_old(Tensor& A, Tensor& B, size_t mode){
    /*
    * SAME AS TENSOR_CONTRACT_ALL_BUT_ONE, but instead, it's batched.
    * Unfolds two tensors A and B in the same mode, and computes A_(k) B_(k)^T, storing the result C
    * in column-major order.
    *   - Instead of manually unfolding these two tensors, we can get "blocks" from each of the buffers, multiply them, 
    *     and then sum the result.
    *   - In CblasColMajor mode, treat A's memory as A^T (num_cols x n_k, lda = n_k)
    *     and B's memory as B (p x n_k, ldb = n).
    *   - We want C (n_k x p, col-major, ldc = m) = A * B^T = (A^T)^T * (B)^T
    *   - So call: C(n_k x p) = Trans(A^T) * Trans(B)
    *     i.e., cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans, n_k, , k,
    *                       1.0, A, k, B, n, 0.0, C, m)
    */
    
    size_t i = 0; 

    // Take into account the mode which both are unfolding for. Remember this is A(k) (B(k))^T
    // The dimensions of the final tensor will be n_k x p. 
    // Get necessary dimensions
    std::vector<size_t> A_dims = A.getdims();
    std::vector<size_t> B_dims = B.getdims();

    size_t n_k = A_dims[mode];
    size_t p = B_dims[mode];

    // Determine the number of columns for each block and set dimensions accordingly. 
    size_t num_cols = 1;
    for (i = 0; i < mode; ++i) { 
      assert(A_dims[i] == B_dims[i] && "Dimensions of A and B must be equal except at n_k.");
      num_cols *= A_dims[i];
    }

    // This will be a cumulative matrix, makes the buffer all zeros to start.
    Matrix C(p*n_k, n_k, p);
    std::memset(C.data_ptr, 0, C.buflen * sizeof(double));

    // Get the number of "blocks" we are multiplying.
    size_t num_blocks = 1;
    for (i = mode + 1; i < A.ndim; ++i){ 
      assert(A_dims[i] == B_dims[i] && "Dimensions of A and B must be equal except at n_k.");
      num_blocks *= A_dims[i];
    }

    // Matrix multiply each block and sum in C.
    // A blocks have dimensions n_k X num_cols, B blocks have dimensions p X num_cols, hence why we transpose B by default.
    // std::cout << "p = " << p << "\nn_k = " << n_k << "\nnum_blocks = " << num_blocks << "\nnum_cols = " << num_cols << std::endl;
    // MKL Strided Batched BLAS documentation: 
    // https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/cblas-gemm-batch-strided.html
    MKL_INT cblas_m = (MKL_INT) n_k;
    MKL_INT cblas_k = (MKL_INT) num_cols;
    MKL_INT cblas_n = (MKL_INT) p;
    double cblas_alpha = 1.0;
    double cblas_beta = 1.0;
    double* cblas_a = A.data_ptr;
    MKL_INT cblas_lda = (MKL_INT) num_cols;
    MKL_INT cblas_stridea = (MKL_INT) (n_k * num_cols);
    double* cblas_b = B.data_ptr;
    MKL_INT cblas_ldb = (MKL_INT) num_cols;
    MKL_INT cblas_strideb = (MKL_INT) (p * num_cols);
    double* cblas_c = C.data_ptr; 
    MKL_INT cblas_ldc = (MKL_INT) n_k;
    MKL_INT cblas_stridec = (MKL_INT) 0;
    MKL_INT cblas_batch_size = (MKL_INT) num_blocks;
    
   // Make a for loop here. The "num_threads" we need to test to see what the best value is here. 
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

    // TODO: SUM C MATRICES IN PARALLEL.
    

    return C;
}


Tensor tensor_minus_tensor(Tensor &A, Tensor &B){
  assert(A.getdims() == B.getdims() && "Tensors A and B must have the same dimensions.");
  // Get resulting tensor dimensions.
  std::vector<size_t> TOdims(A.dims);
  
  size_t TObuflen = A.buflen;
  Tensor TO(TObuflen, A.ndim, TOdims);
  std::memset(TO.data_ptr, 0, TObuflen * sizeof(double));

  // Only use openMP fork/join for large tensors. Otherwise the cost isn't worth it.
  #pragma omp parallel for simd schedule(static) if(TObuflen > 100000)
  for(size_t i = 0; i < TObuflen; ++i){
    TO.data_ptr[i] = A.data_ptr[i] - B.data_ptr[i];
  }

  return TO;

}

Tensor tensor_plus_tensor(Tensor &A, Tensor &B, Tensor &C){
  assert(A.getdims() == B.getdims() && "Tensors A and B must have the same dimensions.");
  assert(C.getdims() == B.getdims() && "Tensors B and C must have the same dimensions.");
  assert(C.getdims() == A.getdims() && "Tensors A and C must have the same dimensions.");
  // Get resulting tensor dimensions.
  std::vector<size_t> TOdims(A.dims);

  size_t TObuflen = A.buflen;
  Tensor TO(TObuflen, A.ndim, TOdims);
  std::memset(TO.data_ptr, 0, TObuflen * sizeof(double));

  // Only use openMP fork/join for large tensors. Otherwise the cost isn't worth it.
  #pragma omp parallel for simd schedule(static) if(TObuflen > 100000)
  for(size_t i = 0; i < TObuflen; ++i){
    TO.data_ptr[i] = A.data_ptr[i] + B.data_ptr[i] + C.data_ptr[i];
  }

  return TO;

}

Tensor hadamard_pointwise(Tensor &A, Tensor &B){
  /*
  Multiply two tensors elementwise.
  */
  assert(A.getdims() == B.getdims() && "Tensors A and B must have the same dimensions.");
  // Get resulting tensor dimensions.
  std::vector<size_t> TOdims(A.dims);
  
  size_t TObuflen = A.buflen;
  Tensor TO(TObuflen, A.ndim, TOdims);
  std::memset(TO.data_ptr, 0, TObuflen * sizeof(double));

  // Only use openMP fork/join for large tensors. Otherwise the cost isn't worth it.
  #pragma omp parallel for simd schedule(static) if(TObuflen > 100000)
  for(size_t i = 0; i < TObuflen; ++i){
    TO.data_ptr[i] = A.data_ptr[i] * B.data_ptr[i];
  }

  return TO;

}
Matrix get_slicewise_diagonals(Tensor &A){
  /*
  Return the diagonal values of a tensor as a matrix.
  */
  std::vector<size_t> A_dims = A.getdims();
  size_t M_rows = A_dims[0]; // This will give us k.
  size_t A_cols = A_dims[1];
  Matrix M(M_rows * A.nslices, M_rows, A.nslices);
  std::memset(M.data_ptr, 0, M.buflen * sizeof(double));

  // Only use openMP fork/join for large tensors. Otherwise the cost isn't worth it.
  #pragma omp parallel for simd schedule(static) if(M.buflen > 100000)
  for(size_t i = 0; i < M.buflen; ++i){
    int diag_index = (i % A_cols) + (i * A_cols);
    M.data_ptr[i] = A.data_ptr[diag_index];
  }

  return M;

}
// Matrix matrix_transpose(const Matrix &M){
//   std::vector<size_t> out_dims = [M.dims[1], M.dims[0]]

// }

// ===========================================================================
// Matrix exponential
// ===========================================================================
//
// Two evaluation paths and two retractions are provided.
//
//   Path B (general, and the DEFAULT):  Pade scaling-and-squaring
//                    (`expm_pade`), Higham 2005 Algorithm 2.3.  Direct, no
//                    decomposition, almost pure GEMM.  Also `cayley`, the
//                    [1/1] diagonal Pade retraction, one factor-and-solve.
//
//   Path A (reuse, OPT-IN):  for a skew-symmetric S, one real Schur
//                    decomposition S = Q T Q^T is computed once
//                    (`skew_schur`).  T is exactly block diagonal with 2x2
//                    blocks [[0, theta],[-theta, 0]] (plus one 1x1 zero block
//                    when n is odd), so exp(t*S) = Q*blkdiag(rot(t*theta_j))*Q^T
//                    can then be evaluated for any t (`expm_skew_apply`) at the
//                    cost of an O(n^2) rotation assembly plus a single GEMM.
//                    Cayley reuses the same decomposition through
//                    `cayley_skew_apply`, since Cayley of a 2x2 skew block has
//                    the closed form (1-(a/2)^2)/(1+(a/2)^2), a/(1+(a/2)^2).
//
// WHY PATH B IS THE DEFAULT.  An earlier survey note recommended always
// decomposing.  The adversarial audit (VERIFICATION_audit.md, A6/A7) overturned
// that: the Schur setup is expensive enough that reuse only wins past roughly
// 4-9 trial steps against direct Pade (n-dependent: K = 8.7 at n=500, 6.6 at
// n=1000, 4.3 at n=2000) and past roughly 17 against direct Cayley, while the
// measured Armijo backtracking count is 1-4, averaging under 2.  So
// `expm_multi` makes this a runtime decision against an explicit threshold
// (EXPM_REUSE_MIN_STEPS_*) rather than always decomposing, and the single-shot
// `expm` never decomposes.
//
// ACCURACY.  The audit (A8) also showed that the widely quoted 1e-15 forward
// error gate was an artifact of injecting a random, orthogonality-destroying
// perturbation.  With a structure-preserving error the optimizer is unaffected
// out to ~1e-8.  The binding requirement is therefore that the result stay on
// the orthogonal group; every path here does (diagonal Pade and Cayley are
// exactly orthogonal on skew input in exact arithmetic, and the Schur route is
// orthogonal up to the conditioning of Q).  Forward accuracy is nonetheless
// kept at the natural backward-stable level -- nothing here is deliberately
// loosened.
//
// ---------------------------------------------------------------------------
// Backend split
// ---------------------------------------------------------------------------
// The heavy linear algebra is funnelled through the four thin `expm_be_*`
// entry points below (GEMM, factor-and-solve, skew Schur decomposition, and a
// handful of O(n^2)/O(n) vector helpers).  Everything above them -- the skew
// check, the degree and scaling selection, the squaring loop bookkeeping and
// the rotation assembly -- contains no direct MKL calls.  A GPU backend
// attaches by providing alternative bodies for these functions (cuBLAS dgemm,
// cuSOLVER getrf/getrs); note that cuSOLVER has no real Schur driver, so a
// device build would either keep `expm_be_schur_skew` on the host or fall back
// to Path B, which is almost pure GEMM and therefore the GPU-friendly route.
// The decision logic does not need to change in either case.
// ---------------------------------------------------------------------------

// Backend: C <- alpha * op(A) * op(B) + beta * C, all n-by-n, column major.
static void expm_be_gemm(size_t n, bool transA, bool transB, double alpha,
                         const double* A, const double* B, double beta,
                         double* C) {
  cblas_dgemm(
    CblasColMajor, // Column major order. `Layout` parameter of MKL cblas call.
    transA ? CblasTrans : CblasNoTrans, // `transa` param of MKL cblas call.
    transB ? CblasTrans : CblasNoTrans, // `transb` param of MKL cblas call.
    (lapack_int) n, // Number of rows of A or C. `m` param of MKL cblas call.
    (lapack_int) n, // Number of cols of B or C. `n` param of MKL cblas call.
    (lapack_int) n, // Inner dimension. `k` param of MKL cblas call.
    alpha, // Scalar `alpha` param of MKL cblas call.
    A, // Data buffer of A. `a` param of MKL cblas call.
    (lapack_int) n, // Leading dimension of A. `lda` param of MKL cblas call.
    B, // Data buffer of B. `b` param of MKL cblas call.
    (lapack_int) n, // Leading dimension of B. `ldb` param of MKL cblas call.
    beta, // Scalar `beta` param of MKL cblas call.
    C, // Data buffer of C. `c` param of MKL cblas call.
    (lapack_int) n // Leading dimension of C. `ldc` param of MKL cblas call.
  );
}

// Backend: solve D * X = R in place (D is n-by-n, R is n-by-n, column major).
// D is overwritten by its LU factors and R by the solution.
// Unlike the surrounding kernels in this file, `info` is checked and a failure
// raises.  That is a deliberate departure from the house style: a singular
// Pade denominator silently returning garbage would be indistinguishable from
// a correct answer, and the accuracy gate this kernel exists for is 1e-15.
static void expm_be_factor_and_solve(size_t n, double* D, double* R) {
  lapack_int nn = (lapack_int) n;
  lapack_int info = 0;
  // Padded by 64 entries: at least one LAPACK routine in this build writes
  // past its documented array bounds (see the DBDSVDX pad in svdx above), so
  // every scratch array in this kernel is allocated defensively.
  lapack_int* ipiv = (lapack_int*) malloc((n + 64) * sizeof(lapack_int));

  dgetrf(
    &nn, // M: No. of rows of the input matrix.
    &nn, // N: No. of columns of the input matrix.
    D, // A: Input matrix. Overwritten by the L and U factors.
    &nn, // LDA: Leading dimension of A.
    ipiv, // IPIV: Pivot indices; row i was interchanged with row IPIV(i).
    &info // INFO: Exit code.
  );
  if (info != 0) {
    free(ipiv);
    throw std::runtime_error("expm: dgetrf failed (singular Pade denominator), info = "
                             + std::to_string((long long) info));
  }

  dgetrs(
    "N", // TRANS: 'N' solves D * X = R (no transpose); the data is column major.
    &nn, // N: Order of the matrix D.
    &nn, // NRHS: Number of right hand sides, one per column of R.
    D, // A: The LU factors produced by DGETRF.
    &nn, // LDA: Leading dimension of A.
    ipiv, // IPIV: Pivot indices from DGETRF.
    R, // B: Right hand sides on entry, solution on exit.
    &nn, // LDB: Leading dimension of B.
    &info // INFO: Exit code.
  );
  free(ipiv);
  if (info != 0) {
    throw std::runtime_error("expm: dgetrs failed, info = "
                             + std::to_string((long long) info));
  }
}

// Backend: real Schur decomposition S = Q T Q^T of a skew-symmetric S.
// On exit Q holds the orthogonal factor and T the (block diagonal) Schur form;
// both are caller-allocated n-by-n column major buffers.  S itself is read
// only -- the destructive copy is made here.
//
// This is the one step a GPU backend cannot currently replace (no cuSOLVER
// real Schur driver); it is isolated here for exactly that reason.
static void expm_be_schur_skew(size_t n, const double* S, double* Q, double* T,
                               bool verbose) {
  lapack_int nn = (lapack_int) n;
  lapack_int sdim = 0, info = 0, lwork = -1;
  double wkopt = 0.0;
  size_t pad = 64; // Defensive padding, as in expm_be_factor_and_solve.

  std::copy(S, S + n * n, T); // DGEES overwrites its input with the Schur form.

  std::vector<double> wr(n + pad, 0.0), wi(n + pad, 0.0);
  std::vector<lapack_int> bwork(n + pad, 0); // Not referenced for SORT = 'N',
                                             // allocated anyway.

  // Workspace query
  dgees(
    "V", // JOBVS: 'V' computes the orthogonal Schur vectors Q.
    "N", // SORT: 'N' leaves the eigenvalues unordered.
    (MKL_D_SELECT_FUNCTION_2) NULL, // SELECT: not referenced for SORT = 'N'.
    &nn, // N: Order of the input matrix.
    T, // A: Input matrix, overwritten by the real Schur form.
    &nn, // LDA: Leading dimension of A.
    &sdim, // SDIM: Number of selected eigenvalues; 0 for SORT = 'N'.
    wr.data(), // WR: Real parts of the eigenvalues.
    wi.data(), // WI: Imaginary parts of the eigenvalues.
    Q, // VS: Matrix holding the orthogonal Schur vectors.
    &nn, // LDVS: Leading dimension of VS.
    &wkopt, // WORK: On a query, receives the optimal workspace size.
    &lwork, // LWORK: -1 requests the optimal workspace size.
    bwork.data(), // BWORK: Not referenced for SORT = 'N'.
    &info // INFO: Exit code.
  );
  if (info != 0) {
    throw std::runtime_error("expm: dgees workspace query failed, info = "
                             + std::to_string((long long) info));
  }

  lwork = (lapack_int) wkopt;
  double* work = (double*) malloc((lwork + (lapack_int) pad) * sizeof(double));

  if (verbose) {
    printf("Size of the work array for DGEES: %lld\n", (long long) lwork);
  }

  // Compute the real Schur decomposition
  dgees(
    "V", // JOBVS: 'V' computes the orthogonal Schur vectors Q.
    "N", // SORT: 'N' leaves the eigenvalues unordered.
    (MKL_D_SELECT_FUNCTION_2) NULL, // SELECT: not referenced for SORT = 'N'.
    &nn, // N: Order of the input matrix.
    T, // A: Input matrix, overwritten by the real Schur form.
    &nn, // LDA: Leading dimension of A.
    &sdim, // SDIM: Number of selected eigenvalues; 0 for SORT = 'N'.
    wr.data(), // WR: Real parts of the eigenvalues.
    wi.data(), // WI: Imaginary parts of the eigenvalues.
    Q, // VS: Matrix holding the orthogonal Schur vectors.
    &nn, // LDVS: Leading dimension of VS.
    work, // WORK: Workspace.
    &lwork, // LWORK: Dimension of the array WORK.
    bwork.data(), // BWORK: Not referenced for SORT = 'N'.
    &info // INFO: Exit code.
  );

  free(work);
  if (info != 0) {
    throw std::runtime_error("expm: dgees failed to converge, info = "
                             + std::to_string((long long) info));
  }
}

// Backend: y <- alpha*x + y over N contiguous entries.
static void expm_be_axpy(size_t N, double alpha, const double* x, double* y) {
  cblas_daxpy((lapack_int) N, alpha, x, 1, y, 1);
}

// Backend: x <- alpha*x over N contiguous entries.
static void expm_be_scal(size_t N, double alpha, double* x) {
  cblas_dscal((lapack_int) N, alpha, x, 1);
}

// Backend: plane rotation of two length-n columns,
// (x, y) <- (c*x + s*y, c*y - s*x).
static void expm_be_rot(size_t n, double* x, double* y, double c, double s) {
  cblas_drot((lapack_int) n, x, 1, y, 1, c, s);
}

// Backend: 1-norm (maximum absolute column sum) of an n-by-n column major A.
static double expm_be_norm1(size_t n, const double* A) {
  double nrm = 0.0;
  for (size_t j = 0; j < n; j++) {
    double colsum = cblas_dasum((lapack_int) n, A + j * n, 1);
    if (colsum > nrm) nrm = colsum;
  }
  return nrm;
}

// ---------------------------------------------------------------------------
// Pade constants.  Transcribed from Higham 2005 Table 2.3 and Algorithm 2.3
// (see pade_scaling_squaring.md in the survey notes); all of the b_k below are
// exactly representable as IEEE doubles.
//
// THRESHOLD CONVENTION: this kernel follows Higham 2005 Algorithm 2.3, i.e.
// degree selection on the plain 1-norm ||A||_1 with theta_13 = 5.371920351148152.
// That is the Eigen/Julia convention.  SciPy and MATLAB instead follow
// Al-Mohy & Higham 2009 Algorithm 5.1, which selects on the sharper estimates
// ||A^k||_1^(1/k) and lowers theta_13 to 4.25 (to cut kappa(q_13) from 2.2e2 to
// 7.1e1).  Both satisfy the same backward error bound ||dA||/||A|| <= u = 2^-53;
// AMH 2009 is never less accurate and is sometimes cheaper, at the cost of a
// 1-norm power estimator.  Higham 2005 was chosen here for simplicity and
// because its degree selection is exact rather than estimated.
//
// Two traps avoided here, both from VERIFICATION_audit.md:
//  * No AMH `ell` correction is used (Algorithm 2.3 has none).  If one is ever
//    added, do NOT copy the coeff[] literals out of SciPy's C kernel: three of
//    the five (m = 7, 9, 13) are imprecise.  The exact value is
//    1/|c_{2m+1}| = (2m)!(2m+1)!/(m!)^2, scaled by 2^-53 (audit A5).
//  * The even powers are formed with plain GEMM, not DSYRK.  The symmetric
//    rank-k trick is a ~1.9x win single-threaded but a pessimization at high
//    thread counts (up to 11x slower at n = 2000 on 128 threads), and pystarm
//    targets 192-core nodes (audit A3).
// ---------------------------------------------------------------------------
static const double EXPM_THETA[4] = { // theta_3, theta_5, theta_7, theta_9
  1.495585217958292e-2, 2.539398330063230e-1,
  9.504178996162932e-1, 2.097847961257068e0
};
static const double EXPM_THETA13 = 5.371920351148152e0;

static const double EXPM_B3[4] = {120., 60., 12., 1.};
static const double EXPM_B5[6] = {30240., 15120., 3360., 420., 30., 1.};
static const double EXPM_B7[8] = {17297280., 8648640., 1995840., 277200.,
                                  25200., 1512., 56., 1.};
static const double EXPM_B9[10] = {17643225600., 8821612800., 2075673600.,
                                   302702400., 30270240., 2162160., 110880.,
                                   3960., 90., 1.};
static const double EXPM_B13[14] = {64764752532480000., 32382376266240000.,
                                    7771770303897600., 1187353796428800.,
                                    129060195264000., 10559470521600.,
                                    670442572800., 33522128640., 1323241920.,
                                    40840800., 960960., 16380., 182., 1.};

// C <- cI*I + sum_j c[j] * P[j], with P[j] the even powers A^2, A^4, ...
static void expm_poly_terms(double* C, size_t n, double cI,
                            double* const* P, const double* c, int k) {
  size_t N = n * n;
  std::fill(C, C + N, 0.0);
  for (int j = 0; j < k; j++) expm_be_axpy(N, c[j], P[j], C);
  for (size_t i = 0; i < n; i++) C[i + i * n] += cI;
}

// Returns true when A is skew-symmetric to a relative tolerance, i.e. when
// max_ij |A_ij + A_ji| <= tol * max_ij |A_ij|.  The diagonal is included, so a
// nonzero diagonal entry fails the test.  The all-zero matrix passes.
bool is_skew_symmetric(const Matrix& A, double tol = 1e-12) {
  if (A.nrow != A.ncol) return false;
  size_t n = A.nrow;
  double maxabs = 0.0, maxasym = 0.0;
  for (size_t j = 0; j < n; j++) {
    for (size_t i = j; i < n; i++) {
      double aij = A.data_ptr[i + j * n];
      double aji = A.data_ptr[j + i * n];
      double m = std::max(std::fabs(aij), std::fabs(aji));
      double a = std::fabs(aij + aji);
      if (m > maxabs) maxabs = m;
      if (a > maxasym) maxasym = a;
    }
  }
  return maxasym <= tol * maxabs;
}

// Path A, step one.  Real Schur decomposition of a skew-symmetric S.
//
// Returns (Q, theta) with Q the n-by-n orthogonal Schur factor and theta a
// length-n vector encoding the block angles:  theta[i] != 0 means rows/cols
// (i, i+1) form a 2x2 block [[0, theta[i]], [-theta[i], 0]] of the Schur form,
// and theta[i] == 0 means index i is a 1x1 (zero) block.  DGEES places the 1x1
// zero block of an odd-sized skew matrix automatically, so odd n needs no
// special handling.  A 2x2 block whose angle is exactly zero is indistinguishable
// from two 1x1 zero blocks under this encoding, which is harmless because both
// exponentiate to the identity.
//
// S is read only, by design: the whole point of this routine is that the caller
// keeps S and reuses (Q, theta) across many step sizes.  The destructive copy
// DGEES requires is made internally.
std::tuple<Matrix, std::vector<double>> skew_schur(const Matrix& S,
                                                   bool verbose = false) {
  if (S.nrow != S.ncol) {
    throw std::runtime_error("skew_schur: input matrix must be square");
  }
  size_t n = S.nrow;
  Matrix Q(n * n, n, n);
  std::vector<double> theta(n, 0.0);
  if (n == 0) return std::make_tuple(std::move(Q), std::move(theta));

  // Padded, see the note in expm_be_factor_and_solve.
  double* T = (double*) malloc((n * n + 64) * sizeof(double));
  expm_be_schur_skew(n, S.data_ptr, Q.data_ptr, T, verbose);

  // Read the angles straight off the superdiagonal of the Schur form.  A 2x2
  // block is exactly where the eigenvalue pair is complex, which for a skew
  // matrix is where the block is a rotation generator.
  double maxdiag = 0.0;
  for (size_t i = 0; i < n; ) {
    bool two_by_two = (i + 1 < n) && (T[(i + 1) + i * n] != 0.0);
    if (two_by_two) {
      theta[i] = T[i + (i + 1) * n];
      i += 2;
    } else {
      i += 1;
    }
  }
  for (size_t i = 0; i < n; i++) {
    maxdiag = std::max(maxdiag, std::fabs(T[i + i * n]));
  }

  if (verbose) {
    printf("skew_schur: n = %zu, max |diag(T)| = %e\n", n, maxdiag);
  }

  free(T);
  return std::make_tuple(std::move(Q), std::move(theta));
}

// Path A, step two.  Build f(t*S) = Q * blkdiag(G(t*theta_j)) * Q^T from a
// decomposition produced by skew_schur, where G(a) = [[c, s], [-s, c]] is the
// 2x2 rotation supplied by the retraction:
//   exponential: c = cos(a),                      s = sin(a)
//   Cayley:      c = (1-(a/2)^2)/(1+(a/2)^2),     s = a/(1+(a/2)^2)
// Cost is n/2 plane rotations (O(n^2) total) plus one GEMM, independent of t,
// which is what makes a reused decomposition cheap per trial step.
static Matrix skew_blockdiag_apply(const Matrix& Q,
                                   const std::vector<double>& theta,
                                   double t, bool use_cayley, bool verbose,
                                   const char* who) {
  if (Q.nrow != Q.ncol) {
    throw std::runtime_error(std::string(who) + ": Q must be square");
  }
  if (theta.size() != Q.nrow) {
    throw std::runtime_error(std::string(who) + ": theta must have length n");
  }
  size_t n = Q.nrow;
  Matrix X(n * n, n, n);
  if (n == 0) return X;

  // W = Q * D(t).  Column pair (i, i+1) of Q is rotated by G(t*theta_i):
  // D has [[c, s], [-s, c]] there, so W[:,i] = c*Q[:,i] - s*Q[:,i+1] and
  // W[:,i+1] = s*Q[:,i] + c*Q[:,i+1], which is one DROT with (c, -s).
  Matrix W(n * n, n, n);
  std::copy(Q.data_ptr, Q.data_ptr + n * n, W.data_ptr);
  size_t nblocks = 0;
  for (size_t i = 0; i < n; ) {
    if ((i + 1 < n) && (theta[i] != 0.0)) {
      double a = t * theta[i];
      double c, s;
      if (use_cayley) {
        double h = 0.5 * a;
        double d = 1.0 + h * h;
        c = (1.0 - h * h) / d;
        s = a / d;
      } else {
        c = std::cos(a);
        s = std::sin(a);
      }
      expm_be_rot(n, W.data_ptr + i * n, W.data_ptr + (i + 1) * n, c, -s);
      nblocks++;
      i += 2;
    } else {
      i += 1; // 1x1 zero block: the rotation is the identity, nothing to do.
    }
  }

  // X = W * Q^T
  expm_be_gemm(n, false, true, 1.0, W.data_ptr, Q.data_ptr, 0.0, X.data_ptr);
  W.clear();

  if (verbose) {
    printf("%s: n = %zu, t = %e, %zu rotation blocks\n", who, n, t, nblocks);
  }

  return X;
}

// exp(t*S) from a skew_schur decomposition.
Matrix expm_skew_apply(const Matrix& Q, const std::vector<double>& theta,
                       double t = 1.0, bool verbose = false) {
  return skew_blockdiag_apply(Q, theta, t, false, verbose, "expm_skew_apply");
}

// cayley(t*S) from a skew_schur decomposition.
Matrix cayley_skew_apply(const Matrix& Q, const std::vector<double>& theta,
                         double t = 1.0, bool verbose = false) {
  return skew_blockdiag_apply(Q, theta, t, true, verbose, "cayley_skew_apply");
}

// Path A, one shot.  exp(t*A) for a skew-symmetric A, decomposing and applying
// in one call.  Provided mainly as a correctness oracle and for callers who
// know they want this route: for a single t it is several times slower than
// expm_pade.  Use skew_schur plus expm_skew_apply when many t share one A.
Matrix expm_skew(const Matrix& A, double t = 1.0, bool verbose = false) {
  Matrix Q;
  std::vector<double> theta;
  std::tie(Q, theta) = skew_schur(A, verbose);
  Matrix X = expm_skew_apply(Q, theta, t, verbose);
  Q.clear();
  return X;
}

// Path B.  exp(t*A) for a general square A by Pade scaling and squaring,
// Higham 2005 Algorithm 2.3.  No Ward preprocessing (shift and balance) is
// applied, matching Eigen's choice.
//
// A is read only; the scaled working copy is made internally.
Matrix expm_pade(const Matrix& A, double t = 1.0, bool verbose = false) {
  if (A.nrow != A.ncol) {
    throw std::runtime_error("expm_pade: input matrix must be square");
  }
  size_t n = A.nrow;
  size_t N = n * n;
  size_t pad = 64; // Defensive padding, see expm_be_factor_and_solve.
  Matrix X(N, n, n);
  if (n == 0) return X;

  // B = t*A, the working copy.
  double* B = (double*) malloc((N + pad) * sizeof(double));
  for (size_t i = 0; i < N; i++) B[i] = t * A.data_ptr[i];

  double nrm = expm_be_norm1(n, B);
  if (!(nrm > 0.0)) {
    // Zero (or NaN-free zero) input: exp(0) = I.  Also catches n-by-n zeros,
    // for which the degree selection below would be a waste.
    std::fill(X.data_ptr, X.data_ptr + N, 0.0);
    for (size_t i = 0; i < n; i++) X.data_ptr[i + i * n] = 1.0;
    free(B);
    return X;
  }

  // --- degree and scaling selection (no MKL here) ---
  int m = 13;
  int s = 0;
  const double* b = EXPM_B13;
  if (nrm <= EXPM_THETA[0])      { m = 3;  b = EXPM_B3; }
  else if (nrm <= EXPM_THETA[1]) { m = 5;  b = EXPM_B5; }
  else if (nrm <= EXPM_THETA[2]) { m = 7;  b = EXPM_B7; }
  else if (nrm <= EXPM_THETA[3]) { m = 9;  b = EXPM_B9; }
  else {
    s = (int) std::ceil(std::log2(nrm / EXPM_THETA13));
    if (s < 0) s = 0;
    if (s > 0) expm_be_scal(N, std::ldexp(1.0, -s), B); // exact, power of two
  }

  if (verbose) {
    printf("expm_pade: n = %zu, ||t*A||_1 = %e, degree m = %d, squarings s = %d\n",
           n, nrm, m, s);
  }

  // --- even powers ---
  // npow = number of even powers A^2, A^4, ... needed.
  int npow = (m == 13) ? 3 : (m - 1) / 2;
  double* P[4] = {NULL, NULL, NULL, NULL}; // A^2, A^4, A^6, A^8
  for (int j = 0; j < npow; j++) P[j] = (double*) malloc((N + pad) * sizeof(double));
  double* U   = (double*) malloc((N + pad) * sizeof(double));
  double* V   = (double*) malloc((N + pad) * sizeof(double));
  double* Wk  = (double*) malloc((N + pad) * sizeof(double));
  double* tmp = (double*) malloc((N + pad) * sizeof(double));

  if (npow >= 1) expm_be_gemm(n, false, false, 1.0, B,    B,    0.0, P[0]); // A^2
  if (npow >= 2) expm_be_gemm(n, false, false, 1.0, P[0], P[0], 0.0, P[1]); // A^4
  if (npow >= 3) expm_be_gemm(n, false, false, 1.0, P[0], P[1], 0.0, P[2]); // A^6
  if (npow >= 4) expm_be_gemm(n, false, false, 1.0, P[1], P[1], 0.0, P[3]); // A^8

  if (m == 13) {
    // Higham 2005 Algorithm 2.3 lines 16-20.  The term ordering below is the
    // published one (approximately increasing order of norm); do not reorder.
    const double c_odd_hi[3]  = {b[9],  b[11], b[13]}; // of A^2, A^4, A^6
    const double c_odd_lo[3]  = {b[3],  b[5],  b[7]};
    const double c_even_hi[3] = {b[8],  b[10], b[12]};
    const double c_even_lo[3] = {b[2],  b[4],  b[6]};

    // U = B * (A^6 * (b13*A^6 + b11*A^4 + b9*A^2) + b7*A^6 + b5*A^4 + b3*A^2 + b1*I)
    expm_poly_terms(tmp, n, 0.0,    P, c_odd_hi, 3);
    expm_poly_terms(Wk,  n, b[1],   P, c_odd_lo, 3);
    expm_be_gemm(n, false, false, 1.0, P[2], tmp, 1.0, Wk); // Wk += A^6 * tmp
    expm_be_gemm(n, false, false, 1.0, B, Wk, 0.0, U);

    // V = A^6 * (b12*A^6 + b10*A^4 + b8*A^2) + b6*A^6 + b4*A^4 + b2*A^2 + b0*I
    expm_poly_terms(tmp, n, 0.0,    P, c_even_hi, 3);
    expm_poly_terms(V,   n, b[0],   P, c_even_lo, 3);
    expm_be_gemm(n, false, false, 1.0, P[2], tmp, 1.0, V);
  } else {
    // Scheme (2.10) for m <= 9:
    //   U = B * (b1*I + b3*A^2 + b5*A^4 + ...),  V = b0*I + b2*A^2 + b4*A^4 + ...
    double c_odd[4], c_even[4];
    for (int j = 0; j < npow; j++) {
      c_odd[j]  = b[2 * j + 3];
      c_even[j] = b[2 * j + 2];
    }
    expm_poly_terms(Wk, n, b[1], P, c_odd,  npow);
    expm_poly_terms(V,  n, b[0], P, c_even, npow);
    expm_be_gemm(n, false, false, 1.0, B, Wk, 0.0, U);
  }

  // --- solve ---
  // r_m = (V - U)^-1 (V + U) = I + 2 (V - U)^-1 U, the form SciPy ships: it
  // avoids forming V + U and the cancellation that goes with it.
  expm_be_axpy(N, -1.0, U, V);              // V <- V - U  (the denominator)
  std::copy(U, U + N, X.data_ptr);          // right hand side
  expm_be_factor_and_solve(n, V, X.data_ptr);
  expm_be_scal(N, 2.0, X.data_ptr);
  for (size_t i = 0; i < n; i++) X.data_ptr[i + i * n] += 1.0;

  // --- squaring ---
  double* x = X.data_ptr;
  double* y = tmp;
  for (int i = 0; i < s; i++) {
    expm_be_gemm(n, false, false, 1.0, x, x, 0.0, y);
    std::swap(x, y);
  }
  X.data_ptr = x; // Both buffers came from malloc, so either may be kept.
  tmp = y;

  for (int j = 0; j < npow; j++) free(P[j]);
  free(U);
  free(V);
  free(Wk);
  free(tmp);
  free(B);

  return X;
}

// The Cayley transform cay(t*A) = (I - (t/2)A)^-1 (I + (t/2)A), the [1/1]
// diagonal Pade approximant to exp(t*A).  For skew A the result is exactly
// orthogonal in exact arithmetic and, per the survey probe, is marginally
// better conditioned for orthogonality than the exponential at large steps.
// It agrees with exp(t*A) only to first order -- the gap grows like a^3/12 --
// so it is a different (but equally valid) retraction, not an approximation to
// be used where the exponential is meant.
//
// Cost is one factor-and-solve, roughly (8/3)n^3, against the exponential's
// (7.33 + s)*2n^3.
//
// A is read only.
Matrix cayley(const Matrix& A, double t = 1.0, bool verbose = false) {
  if (A.nrow != A.ncol) {
    throw std::runtime_error("cayley: input matrix must be square");
  }
  size_t n = A.nrow;
  size_t N = n * n;
  Matrix X(N, n, n);
  if (n == 0) return X;

  // D = I - (t/2)A (overwritten by its LU factors), R = I + (t/2)A.
  double* D = (double*) malloc((N + 64) * sizeof(double));
  double h = 0.5 * t;
  for (size_t i = 0; i < N; i++) {
    D[i] = -h * A.data_ptr[i];
    X.data_ptr[i] = h * A.data_ptr[i];
  }
  for (size_t i = 0; i < n; i++) {
    D[i + i * n] += 1.0;
    X.data_ptr[i + i * n] += 1.0;
  }

  expm_be_factor_and_solve(n, D, X.data_ptr);
  free(D);

  if (verbose) printf("cayley: n = %zu, t = %e\n", n, t);
  return X;
}

// Reuse crossover thresholds: the number of trial step sizes at which paying
// for one real Schur decomposition starts to beat evaluating directly at each
// step.  VERIFICATION_audit.md A6 measured 4.3-8.7 against direct Pade and
// about 17 against direct Cayley, under OpenBLAS.  Re-measured here against
// MKL on this machine (n = 200..2000, 1/8/32 threads, shared node so the
// numbers are noisy): K(Pade) came out 0.8-8.4 and K(Cayley) 4-177, i.e. the
// Cayley crossover is considerably further out under MKL than the audit's
// OpenBLAS figure.  The defaults below are set so the reuse path engages only
// when it is clearly worth it; both are overridable per call because the
// crossover is strongly n-, thread- and BLAS-dependent.
static const size_t EXPM_REUSE_MIN_STEPS_EXPM = 8;
static const size_t EXPM_REUSE_MIN_STEPS_CAYLEY = 32;

// Retraction selector for expm_multi.
enum Retraction { RETRACT_EXPM = 0, RETRACT_CAYLEY = 1 };

// Single shot.  exp(t*A) for any square A, by Pade scaling-and-squaring.
// Deliberately does NOT decompose, even for skew input: for one evaluation,
// direct Pade is the cheaper route (audit A6).
//
// A is read only.
Matrix expm(const Matrix& A, double t = 1.0, bool verbose = false) {
  return expm_pade(A, t, verbose);
}

// Many step sizes, one direction.  Evaluates the chosen retraction of t*A for
// every t in `ts`, and makes the reuse decision at run time:
//
//   * if A is skew-symmetric to `skew_tol` AND ts.size() >= the crossover
//     threshold, decompose once with skew_schur and apply per step;
//   * otherwise evaluate each step directly.
//
// `reuse_min_steps = 0` selects the built-in threshold for the chosen
// retraction; pass a positive value to override, or a huge value to disable
// reuse entirely.
//
// A is read only.  Every returned Matrix is owned by the caller and must be
// cleared individually.
std::vector<Matrix> expm_multi(const Matrix& A, const std::vector<double>& ts,
                               int retraction = RETRACT_EXPM,
                               size_t reuse_min_steps = 0,
                               double skew_tol = 1e-12,
                               bool verbose = false) {
  if (A.nrow != A.ncol) {
    throw std::runtime_error("expm_multi: input matrix must be square");
  }
  if (retraction != RETRACT_EXPM && retraction != RETRACT_CAYLEY) {
    throw std::runtime_error("expm_multi: unknown retraction");
  }
  bool use_cayley = (retraction == RETRACT_CAYLEY);

  if (reuse_min_steps == 0) {
    reuse_min_steps = use_cayley ? EXPM_REUSE_MIN_STEPS_CAYLEY
                                 : EXPM_REUSE_MIN_STEPS_EXPM;
  }

  bool skew = is_skew_symmetric(A, skew_tol);
  bool reuse = skew && (ts.size() >= reuse_min_steps);

  if (verbose) {
    printf("expm_multi: n = %zu, %zu steps, retraction = %s, skew = %d, "
           "reuse = %d (threshold %zu)\n",
           A.nrow, ts.size(), use_cayley ? "cayley" : "expm",
           (int) skew, (int) reuse, reuse_min_steps);
  }

  std::vector<Matrix> out;
  out.reserve(ts.size());

  if (reuse) {
    Matrix Q;
    std::vector<double> theta;
    std::tie(Q, theta) = skew_schur(A, verbose);
    for (size_t i = 0; i < ts.size(); i++) {
      out.push_back(skew_blockdiag_apply(Q, theta, ts[i], use_cayley, verbose,
                                         "expm_multi"));
    }
    Q.clear();
  } else {
    for (size_t i = 0; i < ts.size(); i++) {
      out.push_back(use_cayley ? cayley(A, ts[i], verbose)
                               : expm_pade(A, ts[i], verbose));
    }
  }

  return out;
}

#endif
