import pyttb as ttb
import numpy as np

# Algorithm 2 from https://www.pnas.org/doi/epdf/10.1073/pnas.2015851118
def tsvdm_3way(A, M):
    assert(len(A.shape) == 3)
    assert(M.shape[0] == M.shape[1])
    assert(A.shape[2] == M.shape[0])

    m = A.shape[0]
    n = A.shape[1]
    p = A.shape[2]

    Ahat = A.ttm(M, 2) # TTM on the third mode, significance of 3-1=2: 3 from 3rd mode, -1 from zero based intexing

    # To collect slice-wise SVD outputs
    Us = []
    Ss = []
    VTs = []
    for i in range(p): # Iterate over the frontal slices (slices along 3rd mode)
        U, S, VT = np.linalg.svd(Ahat.data[:,:,i])
        Us.append(U)
        Ss.append(np.diag(S))
        VTs. append(VT)
        # print("U", U.shape)
        # print("S", S.shape)
        # print("VT", VT.shape)

    # Each SVD output generated seperately
    # Stacking along the 3rd tensor dimension
    Uhat = ttb.tensor(np.stack(Us, axis=2))
    Shat = ttb.tensor(np.stack(Ss, axis=2))
    VThat = ttb.tensor(np.stack(VTs, axis=2))
    
    # Invert M
    Minv = np.linalg.inv(M)
    
    # TTM on the 3rd mode
    U = Uhat.ttm(Minv, 2)
    S = Shat.ttm(Minv, 2)
    VT = VThat.ttm(Minv, 2)

    # print(Uhat.shape)

    return U, S, VT
