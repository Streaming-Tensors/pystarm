import os
import argparse
import pyttb as ttb
import numpy as np
from alg import tsvdm_3way

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--m", type=int, help="Dimension on 1st mode")
    parser.add_argument("-n", "--n", type=int, help="Dimension on 2nd mode")
    parser.add_argument("-p", "--p", type=int, help="Dimension on 3rd mode")
    args = parser.parse_args()
    print(args)

    m = args.m
    n = args.n
    p = args.p
    A = ttb.tenrand((m,n,p))
    # print(A)

    dftmtx = np.fft.fft(np.eye(p))
    # print(dftmtx)
    
    U, S, VT = tsvdm_3way(A, dftmtx)
    print(U.shape)
    print(S.shape)
    print(VT.shape)
    

    # print(dftmtx.__array_interface__)

