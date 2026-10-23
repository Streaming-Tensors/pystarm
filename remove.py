'''
This is just used to run tests. Will be deleted before final push. 
'''

import numpy as np
from tests.alg import tensor_contract_multiply, tensor_contract_batched, tensor_contraction_product_ttb, contegra_tensor_contract_timed
import pystarm
import numpy as np

modes = [0, 1, 2, 3]
flags = []
n_k = 50
p = 50
m = 50
for mode in modes:
    
    # For testing integrity, make sure all dims except n_k and p are equal.
    A_nelm = n_k * (m ** (len(modes) - 1))
    A_dims = [m for i in modes]
    A_dims[mode] = n_k

    B_nelm =  p *  (m ** (len(modes) - 1))
    B_dims = [m for i in modes]
    B_dims[mode] = p

    A = np.random.rand(A_nelm).reshape(A_dims, order='F')
    B = np.random.rand(B_nelm).reshape(B_dims, order='F')

    C, time_taken = contegra_tensor_contract_timed(A, B, mode)

    print(
        f"Time taken for mode {mode} for tensors of size {A_dims} and {B_dims}: {time_taken}"
    )
