# pystarm

### Assumtions
- All matrices and tensor data would be 64-bit floating point
- Ownership of all data is delegated to C++ side. Hence all pystarm objects need to be explicitly cleared by calling appropriate clear function of the respective class
- Matrices and tensors returned by any function of pystarm would be in Python buffertype.
    - Explicit datastructure preparation as numpy array is needed from the buffer
    - The returned buffer would not have any information on strides
    - Dimension of the matrix and tensor can be figured out by calling respective getdim function. 
    - Reshaping the numpy array without any data movement is possible with the obtained dimension information
    - All data is stored in column major order (Fortran order)

### Dependency
- pybind11
- numpy
- MKL

### Directory structure
`cpp` directory contains all the cpp codes which are wrapped with Python layer to expose the computations as Python package
- `matrix.cpp` contains the matrix class implementation
- `tensor.cpp` contains the tensor class implementation
- `ops.cpp` contains implementation of necessary matrix and tensor operations
- `starm.cpp` contains definition of Python bindings of C++ implementations of data structures and operations

### Build
- Bulding with Intel compiler resulted in errors which is yet to be figured out
- In experiments, we built with default GNU compiler avilable on NERSC Perlmutter
- Set `MKLROOT` if it is already not specified. 
    - In our experiments, we used NERSC Perlmutter, where it can be specified as `export MKLROOT=/global/common/software/nersc9/intel/oneapi/mkl/2024.1`
- Build by running `make all`

### Usage
`test.py` contains unit tests for basic modules.
It can be referred to for the usage. 
For example, the usage for matrix multiplication can be following - 

```
arr1 = np.arange(12, dtype=np.float64).reshape((4,3), order='F')
arr2 = np.arange(12, dtype=np.float64).reshape((3,4), order='F')
mat1 = pystarm.Matrix(arr1, 4, 3)
mat2 = pystarm.Matrix(arr2, 3, 4)
mat3 = pystarm.matmul(mat1, mat2)
arr3 = np.frombuffer(mat3, dtype=np.float64).reshape(mat3.getdims(), order='F', copy = False)
print(arr3)
```
