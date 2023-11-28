######################################################################## imports
#######################################################################

import numpy as np
import re
import math
import time
from generateDeficientMatrix import generate_matrix
from operator import itemgetter

############################################################### global variables
###############################################################

decimalPlaces = 5 # change this to get numpy to show you more or less digits
np.set_printoptions(precision=decimalPlaces)

#################################################################### subroutines
####################################################################

def truncateNumber(num, precision):
    '''
    Truncates a number to the number of decimal places specified and
    returns the number as a string. If truncation cannot be done, the original
    number is returned as a string.
    Inputs:
        num: number to be truncated
        precision: number of decimal places desired
    Outputs:
        strNum: truncated version of number, now a string
    '''

    # convert num to string
    strNum = str(num)

    # number has decimal point and is in scientific notation
    scientific = "e" in strNum and "." in strNum
    # number has decimal point, not in scientific notation
    decimal = "e" not in strNum and "." in strNum

    # find indices of e's and .'s and truncate
    if scientific:
        pointIndex = strNum.index(".")
        eIndex = strNum.index("e")
        decimals = len(strNum[pointIndex:eIndex])
        # make sure we can actually truncate
        if precision > decimals:
            return strNum
        return strNum[0 : pointIndex + precision + 1] + strNum[eIndex :]
    elif decimal:
        pointIndex = strNum.index(".")
        decimals = len(strNum[pointIndex:]) - 1
        # make sure we can actually truncate
        if precision > decimals:
            return strNum
        return strNum[0 : pointIndex + precision + 1]
    else:
        # nothing to truncate
        return strNum
    
def prettyPrintFactorization(A: np.array, B: np.array, C: np.array,\
                             D: np.array, Aname="A", Bname="B", Cname="C",\
                                Dname="D", precision=decimalPlaces):
    '''
    Pretty print the matrix factorization A = B * C * D with included sizes and 
    optional matrix names.
    
    Print format is below:
    [A_11 A_12 ... A_1n]   
    [A_21 A_22 ... A_2n] = 
    [ .    .    .   .  ]
    [A_m1 A_m2 ... A_mn]
                   Aname
                     m x n

    [B_11 B_12 ... B_1n]   [C_11 C_12 ... C_1n]   [D_11 D_12 ... D_1n]
    [B_21 B_22 ... B_2n] * [C_21 C_22 ... C_2n] * [D_21 D_22 ... D_2n]
    [ .    .    .   .  ]   [ .    .    .   .  ]   [ .    .    .   .  ]
    [B_n1 B_n2 ... B_nn]   [C_n1 C_n2 ... C_nn]   [D_n1 D_n2 ... D_nn]
                   Bname                  Cname                  Dname
                     m x n                  m x n                  m x n
    
    Inputs:
        A, B, C, D: matrices
        Aname, Bname, Cname, Dname: (optional) names for the printed matrices
        precision: (optional) number of decimal places to display for matrix 
                   entries, default is decimalPlaces variable
    '''

    # get sizes, (m x n), of A, B, C, D
    Am, An = A.shape[0], A.shape[1]
    Bm, Bn = B.shape[0], B.shape[1]
    Cm, Cn = C.shape[0], C.shape[1]
    Dm, Dn = D.shape[0], D.shape[1]

    # figure out which row to print operators "=" and "*"'s
    maxRowSize = max(Am, Bm, Cm, Dm)
    operatorRow = math.floor(maxRowSize / 2)
    # print(maxRowSize)
    # print(operatorRow)

    # find element with max length of each matrix
    maxElementLengthA = max([len(truncateNumber(element, precision))\
                             for row in A for element in row])
    maxElementLengthB = max([len(truncateNumber(element, precision))\
                             for row in B for element in row])
    maxElementLengthC = max([len(truncateNumber(element, precision))\
                             for row in C for element in row])
    maxElementLengthD = max([len(truncateNumber(element, precision))\
                             for row in D for element in row])
    
    # surprise tool that will help us later
    firstRow = ""

    # - pretty printing time!
    # - iterate over each row value, from [0 to maxRowSize - 1]
    for rowIndex in range(0, maxRowSize):
        ## rows of A matrix
        ##
        
        # make sure A actually has elements at row rowIndex
        if rowIndex >= Am:
            aRow = " "
            aRow = aRow * (An * (maxElementLengthA + 1) + 3)
            if rowIndex == operatorRow:
                aRow += "* "
            else:
                aRow += "  "
            print(aRow, end="")
        else:
            aRow = ""
            # open bracket of A
            aRow += "["
            # iterate over columns of A
            for colIndex in range(0, An):
                # print element, adding spaces as necessary
                Aelement = truncateNumber(A[rowIndex][colIndex], precision)
                while len(Aelement) < maxElementLengthA:
                    Aelement = " " + Aelement
                aRow += Aelement
                # print spaces between elements, sometimes
                if colIndex != An - 1:
                    aRow += " "
            # close bracket of A
            aRow += "] "
            # add equals sign if in correct row
            if rowIndex == operatorRow:
                aRow += "= "
            else:
                aRow += "  "
            print(aRow, end="")

        ## rows of B matrix
        ##

        # make sure B actually has elements at row rowIndex
        if rowIndex >= Bm:
            bRow = " "
            bRow = bRow * (Bn * (maxElementLengthB + 1) + 3)
            if rowIndex == operatorRow:
                bRow += "* "
            else:
                bRow += "  "
            print(bRow, end="")
        else:
            bRow = ""
            # open bracket of B
            bRow += "["
            # iterate over columns of B
            for colIndex in range(0, Bn):
                # print element, adding spaces as necessary
                Belement = truncateNumber(B[rowIndex][colIndex], precision)
                while len(Belement) < maxElementLengthB:
                    Belement = " " + Belement
                bRow += Belement
                # print spaces between elements, sometimes
                if colIndex != Bn - 1:
                    bRow += " "
            # close bracket of B
            bRow += "] "
            # add multiplication sign if in correct row
            if rowIndex == operatorRow:
                bRow += "* "
            else:
                bRow += "  "
            print(bRow, end="")

        ## rows of C matrix
        ## 

        # make sure C actually has elements at row rowIndex
        if rowIndex >= Cm:
            cRow = " "
            cRow = cRow * (Cn * (maxElementLengthC + 1) + 2)
            if rowIndex == operatorRow:
                cRow += "* "
            else:
                cRow += "  "
            print(cRow, end="")
        else:
            cRow = ""
            # open bracket of C
            cRow += "["
            # iterate over columns of C
            for colIndex in range(0, Cn):
                # make sure B actually has elements at row rowIndex
                if rowIndex >= Cm:
                    continue
                # print element, adding spaces as necessary
                Celement = truncateNumber(C[rowIndex][colIndex], precision)
                while len(Celement) < maxElementLengthC:
                    Celement = " " + Celement
                cRow += Celement
                # print spaces between elements, sometimes
                if colIndex != Cn - 1:
                    cRow += " "
            # close bracket of C
            cRow += "] "
            # add multiplication sign if in correct row
            if rowIndex == operatorRow:
                cRow += "* "
            else:
                cRow += "  "
            print(cRow, end="")

        ## rows of D matrix
        ##
        
        # make sure D actually has elements at row rowIndex
        if rowIndex >= Dm:
            dRow = " "
            dRow = dRow * (Dn * (maxElementLengthD + 1) + 3)
            print(dRow)
        else:
            dRow = ""
            # open bracket of D
            dRow += "["
            # iterate over columns of D
            for colIndex in range(0, Dn):
                # make sure B actually has elements at row rowIndex
                if rowIndex >= Dm:
                    continue
                # print element, adding spaces as necessary
                Delement = truncateNumber(D[rowIndex][colIndex], precision)
                while len(Delement) < maxElementLengthD:
                    Delement = " " + Delement
                dRow += Delement
                # print spaces between elements, sometimes
                if colIndex != Dn - 1:
                    dRow += " "
            # close bracket of D
            dRow += "] "
            print(dRow)

        # save first row for later use
        if rowIndex == 0:
            firstRow = aRow + bRow + cRow + dRow
        
    # find indices of string where "[" character is from first row
    brackIndices = [m.start() for m in re.finditer("\]", firstRow)]

    ## matrix names
    ##

    # build name row
    matrixNameRow = ""
    while len(matrixNameRow) <= (brackIndices[0] - len(Aname)):
        matrixNameRow += " "
    matrixNameRow += Aname
    while len(matrixNameRow) <= (brackIndices[1] - len(Bname)):
        matrixNameRow += " "
    matrixNameRow += Bname
    while len(matrixNameRow) <= (brackIndices[2] - len(Cname)):
        matrixNameRow += " "
    matrixNameRow += Cname
    while len(matrixNameRow) <= (brackIndices[3] - len(Dname)):
        matrixNameRow += " "
    matrixNameRow += Dname
    print(matrixNameRow)

    ## matrix sizes
    ##

    # build size row
    matrixSizeRow = ""
    while len(matrixSizeRow) < (brackIndices[0] - len(str(Am)) - 1):
        matrixSizeRow += " "
    matrixSizeRow += str(Am) + " x " + str(An)
    while len(matrixSizeRow) < (brackIndices[1] - len(str(Bm)) - 1):
        matrixSizeRow += " "
    matrixSizeRow += str(Bm) + " x " + str(Bn)
    while len(matrixSizeRow) < (brackIndices[2] - len(str(Cm)) - 1):
        matrixSizeRow += " "
    matrixSizeRow += str(Cm) + " x " + str(Cn)
    while len(matrixSizeRow) < (brackIndices[3] - len(str(Dm)) - 1):
        matrixSizeRow += " "
    matrixSizeRow += str(Dm) + " x " + str(Dn)
    print(matrixSizeRow)

    return

def computeSVD(A: np.array):
    '''
    Compute the Singular Value Decomposition of matrix A.
    Inputs:
        A: (m x n) matrix with rank r > 0
    Outputs:
        (P, Sigma, Q^T, duration), where:
            P: (m x r) matrix with orthonormal columns
            singularValues: (1 x r) matrix of A's singular values
            Q^T: (r x n) matrix with orthonormal columns
            duration: time required to compute SVD
    '''

    # compute P, Q^T, and the singular values of A
    start = time.time()
    (P, singularValues, QT) = np.linalg.svd(A, full_matrices=False)
    end = time.time()
    duration = end - start

    return (P, singularValues, QT, duration)

def SVDrankKApproximation(A: np.array, printFactorization=False,\
                          printDuration=False, k=0):
    '''
    Approximates matrix A of rank r with matrix A_k of rank 1 <= k < r using the
    SVD
    Inputs:
        A: matrix to approximate
        printFactorization: (optional) True will result in original 
                            factorization and rank approximation being printed 
                            to terminal, default False
        printDuration: (optional) True will result in time to compute SVD 
                       factorization being printed to terminal, default False
        k: (optional) rank of desired approximation matrix, default value of 0 
           will choose a value for k based on gap sizes between A's singular 
           values
    Outputs:
        (Ak, Pk, Sigmak, QTk) where:
            Ak: approximation of A
            Pk: leftmost k columns of P
            Sigmak: diagonal matrix of k largest singular values of A
            QTk: topmost k rows of QT
    '''

    # compute SVD of A
    P, singularValues, QT, duration = computeSVD(A)
    # create Sigma = diag(sigma_1, ..., sigma_r)
    Sigma = np.diag(singularValues)

    # find r, the rank of A
    r = np.linalg.matrix_rank(A)

    # make sure k is valid
    if k > r:
        print("ERROR in SVDrankKApproximation: k not less than or equal to", \
              "rank of A")
        return(A, None, None, None)
    elif k == 0:
        (kCalc, gapList) = kFromOrderGap(singularValues)
    else:
        kCalc = k
    
    # compute approximation and output it
    Pk = P[:, 0:kCalc]
    Sigmak = Sigma[0:kCalc, 0:kCalc]
    QTk = QT[0:kCalc, :]
    Ak = np.matmul(Pk, Sigmak)
    Ak = np.matmul(Ak, QTk)

    # print SVD duration
    if printDuration:
        print("Time to create SVD factorization:", str(duration), "\n")

    # print results
    if printFactorization:
        # original SVD
        print("SVD factorization of A, r =", str(r))
        prettyPrintFactorization(A, P, Sigma, QT, Bname="P", Cname="Sigma",\
                                Dname="QT")
        # order differences
        print("Order differences between singular values: ", end="")
        for i in range(0, len(gapList)):
            print(truncateNumber(gapList[i][0], decimalPlaces) + " <--|" +\
                  truncateNumber(gapList[i][1], decimalPlaces) + "|--> ",\
                    end="")
            if i == len(gapList) - 1:
                print(truncateNumber(gapList[i][2], decimalPlaces))  
        # print approximation SVD
        print("Approximation Ak,", "k =", str(kCalc))
        prettyPrintFactorization(Ak, Pk, Sigmak, QTk, Aname="Ak", Bname="Pk",\
                                Cname="Sigmak", Dname="QTk")

    # all done
    return(Ak, Pk, Sigmak, QTk)

def kFromOrderGap(singularValues):
    '''
    Given a list of singular values of some matrix A, finds a value of k to use 
    for the approximating matrix A_k by examining the changes in order between 
    consecutive singular values
    Inputs:
        A: (1 x r) matrix of singular values, arranged in decreasing order
    Outputs:
        k: rank to use for Ak
        orderDiffList: list of differences of order between singular values, 
        elements are tuples of the form (sigma_i, orderDiff, sigma_(i + 1))
    '''

    # find max difference of order
    orderDiffList = [(singularValues[i], np.log10(singularValues[i]) -\
                     np.log10(singularValues[i + 1]), singularValues[i + 1])\
                        for i in range(0, len(singularValues) - 1)]
    maxOrderDiff = max(orderDiffList, key=itemgetter(1))
    # index determines k value
    k = orderDiffList.index(maxOrderDiff) + 1
    
    return (k, orderDiffList)

######################################################################### driver
#########################################################################

def driver():

    ## example usage
    ##


    # create a random (m x n) matrix A with rank deficiency rDef
    # - create an m x n matrix A with integer elements randomly selected from a 
    #   uniform distribution on the interval [a, b]
    m = 7
    n = 3
    rDef = 1
    A = generate_matrix(m, n, rDef)

    
    # - choose an approximating rank
    # - if k = 0, a suitable value for k will be calculated automatically later
    k = 0

    # - create Ak, the rank k approximation of A
    # - the matrices used to construct it can be stored as well
    # - SVDrankKApproximation will not work if k is greater than the rank of A,
    #   so k must be less than or equal to both m and n
    # - decrease the decimalPlaces variable at the top of this file if the
    #   terminal output looks messy
    Ak, Pk, Sigmak, QTk = SVDrankKApproximation(A, printFactorization=True,\
                                                printDuration=True, k=k)

    # TODO: fix findK function so it finds correct gap and not just the largest
    #       one

# - only call driver if this file is run from terminal, prevents driver() from
#   being called if this file is imported into another file
if (__name__ == "__main__"):
    driver()