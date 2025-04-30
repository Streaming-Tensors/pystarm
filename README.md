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

### Build
- `python setup.py build_ext --inplace`

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
