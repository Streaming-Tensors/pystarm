'''
This is just used to run tests. Will be deleted before final push. 
'''

import numpy as np
from tests.alg import tensor_contract_multiply, tensor_contract_batched, tensor_contraction_product_ttb

mode = 2

n_k = 2
p = 2
m = 1
q = 1

np.random.seed(500)
# For testing integrity, make sure all dims except n_k and p are equal.
A_nelm = q * q * m * n_k
A_dims = (q, q, n_k, m)

B_nelm = q * q * m * p
B_dims = (q, q, p, m)

A = np.random.rand(A_nelm).reshape(A_dims, order='F')
B = np.random.rand(B_nelm).reshape(B_dims, order='F')
A = np.asfortranarray(A)
B = np.asfortranarray(B)

# print(f"A = {A}\n\nB = {B}\n\n")

# C_pystarm = tensor_contract_multiply(A, B, mode, naive=False)
C_ttb = tensor_contract_multiply(A, B, mode)
C_pystarm_batched = tensor_contract_batched(A, B, mode)
print(f"C = {C_pystarm_batched}")
print(C_ttb)
    

