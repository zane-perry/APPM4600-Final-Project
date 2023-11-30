######################################################################## imports
#######################################################################

import numpy as np
import re
import math
import time
import scipy as sp

from SVDapproximation import SVDrankKApproximation, kFromOrderGap,\
    prettyPrintFactorization
from QRPermutation import QRrankKApproximation1
from generateDeficientMatrix import generate_matrix

############################################################### global variables
###############################################################

decimalPlaces = 3 # change this to get numpy to show you more or less digits
np.set_printoptions(precision=decimalPlaces)

#################################################################### subroutines
####################################################################

def compareSVDandQR(A, k=0, fullOutput=False):
    '''
    Compare time and error between the SVD and QR approximations of A
    Inputs:
        A: matrix to approximate
        k: (optional, default 0) desired rank of approximation, if 0 is used, 
           a value for k will be determined automatically
        fullOutput: (optional, default False) if True, the factorizations and other
                information will be printed
    Returns:
        (svdTime, qrTime, svdErr, qrErr), where:
            svdTime: time taken for SVD factorization to run
            qrTime: time taken for QR factorization to run
            svdErr: error between difference of SVD approximation and A using 
                    the 2-norm
            qrErr: error between difference of QR approximation and A using the 
                   2-norm
    '''

    # - calculate rank, r, of A
    r = np.linalg.matrix_rank(A)

    # make sure k <= r, otherwise we can't approximate
    if k > r:
        print("ERROR in compareSVDandQR: k not less than or equal to", \
              "rank of A")
        return(0, 0, 0, 0)
    
    # - create the SVD approximation of A and get time required to create the  
    #   SVD of A
    svdK, svdP, svdSigma, svdQT, svdAk, svdPk, svdSigmak, svdQTk, svdTime =\
        SVDrankKApproximation(A, k=k)
    
    # - create the QR approximation of A and get the time required to create the
    #   QR factorization of A
    qrK, qrP, qrQ, qrR, qrAk, qrQk, qrR11, qrR12, qrTime =\
        QRrankKApproximation1(A, k=k)
    qrPT = np.transpose(qrP)
    qrRk = np.block([qrR11, qrR12])

    # compute errors between A and its approximations
    svdErr = np.linalg.norm(A - svdAk, 2)
    qrErr = np.linalg.norm(A - qrAk, 2)

    # TODO: finish output part

    # output
    print("\nTime to compute SVD factorization of A:", str(svdTime))
    print("||A - A_K||_2 =", str(svdErr))
    print("\nTime to compute permuted QR factorization of A:", str(qrTime))
    print("||A - A_k||_2 =", str(qrErr))

    # optional output
    if fullOutput:
        # disclaimer
        print("\nDisplayed precision:", str(decimalPlaces), "decimal places")
        
        # original SVD
        print("\nSVD factorization of A with rank r =", str(r))
        prettyPrintFactorization(A, svdP, svdSigma, svdQT, decimalPlaces,\
                                 Aname="A", Bname="P", Cname="Sigma",\
                                    Dname="Q^T")
        
        # SVD approximation
        print("\nApproximation A_k built from SVD factorization with rank k =",\
              str(svdK))
        prettyPrintFactorization(svdAk, svdPk, svdSigmak, svdQTk,\
                                 decimalPlaces, Aname="A_k", Bname="P_k",\
                                    Cname="Sigma_k", Dname="Q_k^T")
        
        # original QR
        print("\nQR factorization of A with rank r =", str(r))
        prettyPrintFactorization(A, qrQ, qrR, qrPT, decimalPlaces,\
                                 Aname="A", Bname="Q", Cname="R",\
                                    Dname="P^T")
        
        # QR approximation
        print("\nApproximation A_k built from QR factorization with rank k =",\
              str(qrK))
        prettyPrintFactorization(qrAk, qrQk, qrRk, qrP, decimalPlaces,\
                                 Aname="A_k", Bname="Q_k", Cname="R_k",\
                                    Dname="P^T")
        
        print("Q_k @ R_k @ P^T")
        print(qrQk @ qrRk @ qrPT)

    return (svdTime, qrTime, svdErr, qrErr)





    
    
    
def driver():

    # - create a random (m x n) matrix A with rank deficiency rDef
    m = 1000
    n = 1000
    rDef = 760
    A = generate_matrix(m, n, rDef)

    # - choose an approximating rank
    k = 0

    compareSVDandQR(A, k, fullOutput=False)

# - only call driver if this file is run from terminal, prevents driver() from
#   being called if this file is imported into another file
if (__name__ == "__main__"):
    driver()

