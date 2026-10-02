'''
This is just used to run tests. Will be deleted before final push. 
'''

import numpy as np
from tests.alg import tensor_contract_multiply, tensor_contract_batched, tensor_contraction_product_ttb

mode = 3

n_k = 4
p = 2
m = 3
q = 5

np.random.seed(500)
# For testing integrity, make sure all dims except n_k and p are equal.
A_nelm = q * p * m * n_k
A_dims = (q, p, m , n_k)

B_nelm = q * p * m * p
B_dims = (q, p, m, p)

A = np.random.rand(A_nelm).reshape(A_dims, order='F')
B = np.random.rand(B_nelm).reshape(B_dims, order='F')

# print(f"A = {A}")

# C_pystarm = tensor_contract_multiply(A, B, mode, naive=False)
C_ttb = tensor_contraction_product_ttb(A, B, mode)
C_pystarm_batched = tensor_contract_batched(A, B, mode)
print(f"C = {C_pystarm_batched}")
print(C_ttb)
    

