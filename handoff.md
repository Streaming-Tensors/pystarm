Stefan Smith,
July 2026
## Parallel star-M-optimization Handoff Document

In the case where I will be unable to continue my work, I have provided this document to ensure whoever picks up where I left off is not extremely confused :)

Feel free to contact my personal email `stefanosmith2005@gmail.com` with any questions, I'd be more than happy to help!

## Overview
The overall goal of this project is to apply the low $\star_{\mathbf{M}}$ rank approximation objective function and gradient descent algorithm to higher order tensors and make it highly parallel. This way, we can optimize the transformation matrix needed for t-SVDM, improving compression quality. This is done both by using the `pystarm` library and custom kernels for the underlying operations.

### Papers needed for reference and understanding

- $\star_{\mathbf{M}}$-optimization (Newman and Keegan, 2025): https://epubs.siam.org/doi/full/10.1137/24M1702635. This is where you can find the GitHub repo for the MATLAB implementation of the low $\star_{\mathbf{M}}$ rank approximation. It's a good reference, you will probably need it. 
    - Corresponding github: https://github.com/elizabethnewman/star-M-opt

- `pystarm`: https://doi.org/10.48550/arXiv.2605.16058

- If you would like a review of $\star_{\mathbf{M}}$/matrix-mimentic tensor properties as it applies to t-SVDM: https://www.pnas.org/doi/abs/10.1073/pnas.2015851118

### Compiling and running

See `README.md` for instructions on how to compile the library and use it in a python script. If running on ANL's bebop: I've provided additional instructions in `bebop-instructions.md`.

## File organization

There are a lot of files and folders in this, so bear with me here.

### C++ Kernels

Any of the custom/pystarm kernels can be found in `pystarm/cpp`. The custom tensor and matrix class implementations can be found in their respective `.hpp` files, kernels can be found in `ops.cpp`, and the pybind11 API is found in `starm.cpp`. In the TODO section, all kernels that need to be writted/modified should be done so there. You should not modify the `.hpp` files, only the `ops.cpp` and `starm.cpp`. 

### Testing

The python tests used for our all of our kernel implementations are in `tests/test.py`. To run, simply run `python3 tests/test.py`. In the TODO section, all tests that need to be writted should be written here. 

### Parallel-$\star_{\mathbf{M}}$-optimization

This is where all of the optimization occurs, in `star-M-opt`. You can ignore `star_m_product.py`, that's just used for arbitrary testing. The "main" file is `star-M-opt.py`. This calls all the other functions necessary to run the optimization algorithm. 

## TODO!!!

This is an extensive list of what need to be done in order to get this algorithm up and running. This is, to the best of my knowledge, in numerical order. The items at the top should be done first and the items at the bottom should be done last.

### Implementations

- Finish the implementation of the `inv_flag` parameter in `pystarm/cpp/ops.cpp`. It should do elementwise division of a tensor and a diagonal tensor represented as a matrix, in parallel. 
- Write the implementation of `calc_VT_gradient(dVT)`. Refer to Newman and Keegan's code, `tensorSVD/facewiseSVDJacobian.m`.
- Write a matrix exponential retraction kernel. This will be needed before any linesearch is written.
- Write an implementation of Armijo linesearch in `star-M-opt/linesearch.py`. See `optimizers/armijoLinesearch.m` in Newman and Keegan's code.
- Finish writing the implementation of riemannian gradient descent in `star-M-opt.py`, `gradient_descent()`. We are following the same structure of Newman and Keegan's MATLAB code in `optimizers/gradientDescent.m`.

### Test cases

- Test cases for two of the overloaded `slicewise_matmul` funcitons need to be added to `tests/test.py`:
    - One for the one with a tensor and a diagonal tensor represented as a matrix
    - One for the one with two tensors. 
- Fix `starM-product` in `tests/alg.py`, so that the transpose flags work as intended. Right now they don't. See the commented out test case `test_third_order_product_transpose` in `test.py`.
- Once the gradient calculations are written, we will need to find a way to test those gradients to ensure they're correct. This can probably be done with `numpy` or `pyttb`. Just make sure the formulas match those in the supplemental materials of Newman and Keegan's paper.
- Any of the above implementation tasks also need test cases. Write those as well.
- If you notice any operation kernels don't have test cases, feel free to write one. 

At the end of the day, we should be able to pass in an input tensor and return an optimized tensor, where all of the operations are done in parallel. 