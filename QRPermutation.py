import numpy as np
import random
import time
from generateDeficientMatrix import generate_matrix
import matplotlib.pyplot as plt


def driver():

    
    m = random.randint(4,100)
    n = random.randint(4,100)
    minDim = min([m,n])
    r = minDim - random.randint(1,minDim - 3)

    A = generate_matrix(m,n,r)
    

    #A = generate_matrix(100,100,75)


    
    print('QR factorizations for an', m, 'by', n, 'matrix with numerical rank', r)

    [Q,R,duration,diff] = normalQR(A)
    print('Time to create QR factorization:', duration)
    print('Error of ', diff)
    
    [k, P, Q, R, Ak, Q11, R11, R12, duration, diff] = msgQRrankKApproximation(A)
    print('Time to create QR factorization (Modified Gram-Schmidt):', duration)
    print('Rank ', k, 'approximation')
    print('Error of ', diff)
    
    
    [k, P, Q, R, Ak, Q11, R11, R12, duration, diff] = householderQRrankKApproximation(A)
    print('Time to create QR factorization (Householder Reflections):', duration)
    print('Rank ', k, 'approximation')
    print('Error of ', diff)
    

    [k, P, Q, R, Ak, Q11, R11, R12, duration, diff] = givensQRrankKApproximation(A)
    print('Time to create QR factorization (Givens Rotations Modified):', duration)
    print('Rank ', k, 'approximation')
    print('Error of ', diff)


    
    
    

    #createPlots(3,200,50)
    #comparePermuted(201,50)
    

    



def msgQRrankKApproximation(A,k=0,tol=1.e-3):


    if(k==0):
        [Q,R,P,k,duration] = mgsQR(A)
    else:
        [Q,R,P,k,duration] = mgsQR(A,k=k)

    R11 = R[0:k,0:k]
    R12 = R[0:k, k:]
    Q11 = Q[:, 0:k]

    Ak = np.block([Q11 @ R11, Q11 @ R12]) @ np.transpose(P)

    diff = np.linalg.norm(A - Ak)

    return [k, P, Q, R, Ak, Q11, R11, R12, duration, diff]



def householderQRrankKApproximation(A,k=0,tol=1.e-3):

    if(k==0):
        [Q,R,P,k,duration] = householderQR(A,k=k,tol=tol)
    else:
        [Q,R,P,k,duration] = householderQR(A,k=k,tol=tol)

    R11 = R[0:k,0:k]
    R12 = R[0:k, k:]
    Q11 = Q[:, 0:k]

    Ak = np.block([np.matmul(Q11,R11), np.matmul(Q11,R12)])

    diff = np.linalg.norm(A - np.matmul(Ak,np.transpose(P)))


    return [k, P, Q, R, Ak, Q11, R11, R12, duration, diff]


def givensQRrankKApproximation(A,k=0,tol=1.e-3):

    if(k==0):
        [Q,R,P,k,duration] = givensQR(A,k=k,tol=tol)
    else:
        [Q,R,P,k,duration] = givensQR(A,k=k,tol=tol)

    R11 = R[0:k,0:k]
    R12 = R[0:k, k:]
    Q11 = Q[:, 0:k]

    Ak = np.block([np.matmul(Q11,R11), np.matmul(Q11,R12)])

    diff = np.linalg.norm(A - np.matmul(Ak,np.transpose(P)))


    return [k, P, Q, R, Ak, Q11, R11, R12, duration, diff]

def normalQR(A):

    start = time.time()
    R = A.copy()
    [m,n] = R.shape
    Q = np.eye(m)

    for i in range(n):

        if(i == m - 1):
            break
        v = R[i:,i].copy()
        norm = np.linalg.norm(v,ord=2)
        v[0] = v[0] - norm
        H = buildHouseholder(v,m,i)
        R = np.matmul(H,R)
        Q = np.matmul(Q,H)

    end = time.time()
    duration = end-start

    diff = np.linalg.norm(A - np.matmul(Q,R))

    return [Q,R,duration,diff]

def mgsQR(A, k=0, tol=1.e-3):

    start = time.time()
    Q = A.copy()
    forcedRank = True
    [m,n] = Q.shape
    if(k == 0):
        k = n
        forcedRank = False

    foundRank = False

    p = np.array(range(n))

    c = np.zeros(n)

    for j in range(n):
        v = Q[:,j]
        c[j] = np.inner(v,v)

    r = 0

    for i in range(k):
        
        max = i
        maxNorm = c[i]
        for j in range(i + 1, n):
            norm = c[j]
            if(norm > maxNorm):
                max = j
                maxNorm = norm
        if(abs(maxNorm) < tol and not forcedRank):
            foundRank = True
            break
        r += 1
        if(max != i):
            temp = Q[:,i].copy()
            Q[:,i] = Q[:,max]
            Q[:,max] = temp
            p[i], p[max] = p[max], p[i]
        
        ui = Q[:,i]
        ui = ui / np.linalg.norm(ui)
        Q[:,i] = ui

        for k in range(i + 1, n):
            xk = Q[:,k]
            inner = np.inner(ui,xk)
            xk = xk - (inner * ui)
            Q[:,k] = xk
            v = Q[i + 1:,k]
            c[k] = np.inner(v,v)

    if(not foundRank):
        r = min([m,n])

    P = np.zeros([n,n])
    for i in range(n):
        P[p[i],i] = 1

    R = np.matmul(np.matmul(np.transpose(Q),A),P)

    end = time.time()
    duration = end - start
    
    return [Q,R,P,r,duration]



def householderQR(A, k=0, tol=1.e-3):

    start = time.time()
    R = A.copy()
    [m,n] = R.shape
    Q = np.eye(m)
    forcedRank = True

    if(k == 0):
        k = n
        forcedRank = False

    foundRank = False


    p = np.array(range(n))
    c = np.zeros(n)

    for j in range(n):
        v = R[:,j]
        c[j] = np.inner(v,v)

    r = 0

    for i in range(k):

        max = i
        maxNorm = c[i]
        for j in range(i + 1, n):
            norm = c[j]
            if(norm > maxNorm):
                max = j
                maxNorm = norm
        if(abs(maxNorm) < tol and not forcedRank):
            foundRank = True
            break
        if(i == m - 1):
            break
        r += 1
        if(max != i):
            temp = R[:,i].copy()
            R[:,i] = R[:,max]
            R[:,max] = temp
            p[i], p[max] = p[max], p[i]

        v = R[i:,i].copy()
        norm = np.linalg.norm(v,ord=2)
        v[0] = v[0] - norm
        H = buildHouseholder(v,m,i)
        R = np.matmul(H,R)
        Q = np.matmul(Q,H)

        for k in range(i + 1, n):
            v = R[i+1:,k]
            c[k] = np.inner(v,v)

    if(not foundRank):
        r = min([m,n])
    P = np.zeros([n,n])
    for i in range(n):
        P[p[i],i] = 1

    end = time.time()
    duration = end-start

    return [Q,R,P,r,duration]

def buildHouseholder(v,m,i):

    P = 2 * np.outer(v,np.transpose(v)) / (np.inner(v,v))
    Hi = np.eye(m-i) - P
    if(i == 0):
        return Hi
    else:
        return np.block([[np.eye(i), np.zeros([i,m-i])],[np.zeros([m-i,i]), Hi]])
    



def givensQR(A,k=0,tol=1.e-3):


    start = time.time()
    R = A.copy()

    [m,n] = R.shape
    c = np.zeros(n)
    Qt = np.eye(m)
    p = np.array(range(n))

    if(k == 0):
        k = n
        forcedRank = False

    foundRank = False

    for j in range(n):
        v = R[:,j]
        c[j] = np.inner(v,v)

    r = 0

    for i in range(k):
        
        max = i
        maxNorm = c[i]
        for j in range(i + 1, n):
            norm = c[j]
            if(norm > maxNorm):
                max = j
                maxNorm = norm
        if(max != i):
            temp = R[:,i].copy()
            R[:,i] = R[:,max]
            R[:,max] = temp
            p[i], p[max] = p[max], p[i]
        if(abs(maxNorm) < tol and not forcedRank):
            foundRank = True
            break
        if(i == m - 1):
            break
        r += 1

        

        for j in range(i + 1, m):

            G = buildGivens(R[i,i],R[j,i])

            Rrows = np.block([[R[i,i:]],[R[j,i:]]])
            rotatedR = np.matmul(G,Rrows)
            R[i,i:], R[j,i:] = rotatedR

            Qrows = np.block([[Qt[i,:]], [Qt[j,:]]])
            rotatedQ = np.matmul(G,Qrows)
            Qt[i,:], Qt[j,:] = rotatedQ



        for k in range(i+1,n):
            v = R[i+1:,k]
            c[k] = np.inner(v,v)
        

    if(not foundRank):
        r = min([m,n])

    P = np.zeros([n,n])
    for i in range(n):
        P[p[i],i] = 1

    Q = np.transpose(Qt)

    end = time.time()

    duration = end - start

    return [Q,R,P,r,duration]
    


def buildGivens(a,b):

    r = np.hypot(a,b)
    c = a / r
    s = -b / r

    G = np.array([[c,-s],[s,c]])

    return G




def createPlots(a,b,num):

    dim = np.array(range(a,b))
    lengthDim = dim.size  

    mgsTimesDim = np.zeros(lengthDim)
    hhTimesDim = np.zeros(lengthDim)
    gTimesDim = np.zeros(lengthDim)

    mgsErrorsDim = np.zeros(lengthDim)
    hhErrorsDim = np.zeros(lengthDim)
    gErrorsDim = np.zeros(lengthDim)

    rank = np.array(range(1,b))
    lengthRank = rank.size

    mgsTimesRank = np.zeros(lengthRank)
    hhTimesRank = np.zeros(lengthRank)
    gTimesRank = np.zeros(lengthRank)

    mgsErrorsRank = np.zeros(lengthRank)
    hhErrorsRank = np.zeros(lengthRank)
    gErrorsRank = np.zeros(lengthRank)

    for i in range(lengthDim):
        mgsTimeSum = 0
        hhTimeSum = 0
        gTimeSum = 0
        mgsErrorSum = 0
        hhErrorSum = 0
        gErrorSum = 0
        for j in range(num):
            n = dim[i]
            A = generate_matrix(n,n,n)
            print('Dimension ', n, 'Trial', j+1, end='\r')
            duration1, error1 = msgQRrankKApproximation(A)[8:]
            mgsTimeSum += duration1
            mgsErrorSum += error1
            duration2, error2 = householderQRrankKApproximation(A)[8:]
            hhTimeSum += duration2
            hhErrorSum += error2
            duration3, error3 = givensQRrankKApproximation(A)[8:]
            gTimeSum += duration3
            gErrorSum += error3


        mgsTimesDim[i] = mgsTimeSum / num
        hhTimesDim[i] = hhTimeSum / num
        gTimesDim[i] = gTimeSum / num

        mgsErrorsDim[i] = mgsErrorSum / num
        hhErrorsDim[i] = hhErrorSum / num
        gErrorsDim[i] = gErrorSum / num


    for i in range(lengthRank):
        mgsTimeSum = 0
        hhTimeSum = 0
        gTimeSum = 0
        mgsErrorSum = 0
        hhErrorSum = 0
        gErrorSum = 0
        for j in range(num):
            r = rank[i]
            A = generate_matrix(b,b,r)
            print('Rank ', r, 'Trial', j+1, end='\r')
            duration1, error1 = msgQRrankKApproximation(A)[8:]
            mgsTimeSum += duration1
            mgsErrorSum += error1
            duration2, error2 = householderQRrankKApproximation(A)[8:]
            hhTimeSum += duration2
            hhErrorSum += error2
            duration3, error3 = givensQRrankKApproximation(A)[8:]
            gTimeSum += duration3
            gErrorSum += error3


        mgsTimesRank[i] = mgsTimeSum / num
        hhTimesRank[i] = hhTimeSum / num
        gTimesRank[i] = gTimeSum / num

        mgsErrorsRank[i] = mgsErrorSum / num
        hhErrorsRank[i] = hhErrorSum / num
        gErrorsRank[i] = gErrorSum / num


    plt.figure()
    plt.plot(dim,mgsTimesDim, label='Modified Gram-Schmidt')
    plt.plot(dim, hhTimesDim, label='Householder Reflection')
    plt.plot(dim, gTimesDim, label='Givens Rotations')
    plt.xlabel("Size of Matrix")
    plt.ylabel("Time to Factor")
    plt.legend()
    plt.title("Factorization Times for Different Sized Full-Rank Matrices")
    plt.show()

    plt.figure()
    plt.semilogy(dim,mgsErrorsDim, label='Modified Gram-Schmidt')
    plt.semilogy(dim, hhErrorsDim, label='Householder Reflection')
    plt.semilogy(dim, gErrorsDim, label='Givens Rotations')
    plt.xlabel("Size of Matrix")
    plt.ylabel("Error")
    plt.legend()
    plt.title("Errors for Different Sized Full-Rank Matrices")
    plt.show()

    plt.figure()
    plt.plot(rank,mgsTimesRank, label='Modified Gram-Schmidt')
    plt.plot(rank, hhTimesRank, label='Householder Reflection')
    plt.plot(rank, gTimesRank, label='Givens Rotations')
    plt.xlabel("Rank of Matrix")
    plt.ylabel("Time to Factor")
    plt.legend()
    plt.title("Factorization Times for Different Numerical Ranks")
    plt.show()

    plt.figure()
    plt.semilogy(rank,mgsErrorsRank, label='Modified Gram-Schmidt')
    plt.semilogy(rank, hhErrorsRank, label='Householder Reflection')
    plt.semilogy(rank, gErrorsRank, label='Givens Rotations')
    plt.xlabel("Rank of Matrix")
    plt.ylabel("Error")
    plt.legend()
    plt.title("Errors for Different Numerical Ranks")
    plt.show()



def comparePermuted(max,num):

    rank = np.array(range(1,max))
    lengthRank = rank.size

    permutedTimesRank = np.zeros(lengthRank)
    normalTimesRank = np.zeros(lengthRank)

    permutedErrorsRank = np.zeros(lengthRank)
    normalErrorsRank = np.zeros(lengthRank)

    for i in range(lengthRank):
        permutedTimeSum = 0
        normalTimeSum = 0

        permutedErrorSum = 0
        normalErrorSum = 0

        for j in range(num):
            r = rank[i]
            A = generate_matrix(max,max,r)
            print('Rank ', r, 'Trial', j+1, end='\r')
            duration1, error1 = householderQRrankKApproximation(A)[8:]
            permutedTimeSum += duration1
            permutedErrorSum += error1
            duration2, error2 = normalQR(A)[2:]
            normalTimeSum += duration2
            normalErrorSum += error2


        permutedTimesRank[i] = permutedTimeSum / num
        normalTimesRank[i] = normalTimeSum / num

        permutedErrorsRank[i] = permutedErrorSum / num
        normalErrorsRank[i] = normalErrorSum / num


    
    plt.figure()
    plt.plot(rank,normalTimesRank, label='Normal QR')
    plt.plot(rank, permutedTimesRank, label='Permuted QR')
    plt.xlabel("Rank of Matrix")
    plt.ylabel("Time to Factor")
    plt.legend()
    plt.title("Factorization Times for Different Numerical Ranks")
    plt.show()

    plt.figure()
    plt.semilogy(rank,normalErrorsRank, label='Normal QR')
    plt.semilogy(rank, permutedErrorsRank, label='Permuted QR')
    plt.xlabel("Rank of Matrix")
    plt.ylabel("Error")
    plt.legend()
    plt.title("Errors for Different Numerical Ranks")
    plt.show()








# - only call driver if this file is run from terminal, prevents driver() from
#   being called if this file is imported into another file
if (__name__ == "__main__"):
    driver()