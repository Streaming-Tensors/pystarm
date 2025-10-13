from .pystarm import *

def hello(name):
    return f"Hello, {name}!"

def tsvdm_3way(A, M, Minv, k):
    t0 = time.perf_counter()
    A_hat = pystarm.ttm(A, M, len(A.getdims())-1) 
    t1 = time.perf_counter()
    print("[tsvdm_3way] Time for TTM A x M:", t1-t0)

    t0 = time.perf_counter()
    U_hat, S_hat, V_hat = pystarm.slicewise_svdx(A_hat, k) # k=10
    t1 = time.perf_counter()
    print("[tsvdm_3way] Time for slicewise SVD:", t1-t0)

    t0 = time.perf_counter()
    U = pystarm.ttm(U_hat, Minv, len(U_hat.getdims())-1 )
    t1 = time.perf_counter()
    print("[tsvdm_3way] Time for TTM U_hat x Minv:", t1-t0)

    t0 = time.perf_counter()
    S = pystarm.matmul(S_hat, Minv )
    t1 = time.perf_counter()
    print("[tsvdm_3way] Time for matmul S_hat x Minv:", t1-t0)

    t0 = time.perf_counter()
    V = pystarm.ttm(V_hat, Minv, len(U_hat.getdims())-1 )
    t1 = time.perf_counter()
    print("[tsvdm_3way] Time for TTM V_hat x Minv:", t1-t0)
    
    return (U,S,V)

__all__ = [ "hello", "tsvdm_3way" ]
