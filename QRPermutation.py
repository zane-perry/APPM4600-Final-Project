import numpy as np
import random
import scipy as sp
import time
from generateDeficientMatrix import generate_matrix


def driver():


    m = random.randint(4,100)
    n = random.randint(4,100)
    minDim = min([m,n])
    r = minDim - random.randint(1,minDim - 3)

    A = generate_matrix(m,n,r)

    #A = np.array([[16,15,14,13],[12,11,10,9],[8,7,6,5],[4,3,2,1]])
    #A = np.array([[1.,2.,3.,4.],[5.,6.,7.,8.],[9.,10.,11.,12.],[13.,14.,15.,16.]])
    #A = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
    #A = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]])
    #A = np.array([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]])



    print('QR factorizations for an', m, 'by', n, 'matrix with numerical rank', r)
    [k, P, Q, R, Ak, Q11, R11, R12, duration, diff] = msgQRrankKApproximation(A)
    print('Time to create QR factorization (Modified Gram-Schmidt):', duration)
    print('Rank ', k, 'approximation')
    print('Error of ', diff)
    [k, P, Q, R, Ak, Q11, R11, R12, duration, diff] = householderQRrankKApproximation(A)
    print('Time to create QR factorization (Householder Reflections):', duration)
    print('Rank ', k, 'approximation')
    print('Error of ', diff)
    [k, P, Q, R, Ak, Q11, R11, R12, duration, diff] = givensQRrankKApproximation(A)
    print('Time to create QR factorization (Givens Rotations):', duration)
    print('Rank ', k, 'approximation')
    print('Error of ', diff)

    



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
        c[j] = np.matmul(np.transpose(v),v)

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
            c[i], c[max] = c[max], c[i]
        
        ui = Q[:,i]
        ui = ui / np.linalg.norm(ui)
        Q[:,i] = ui

        for k in range(i + 1, n):
            xk = Q[:,k]
            inner = np.inner(ui,xk)
            xk = xk - (inner * ui)
            Q[:,k] = xk
            v = Q[i + 1:,k]
            c[k] = np.matmul(np.transpose(v),v)

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
        c[j] = np.matmul(np.transpose(v),v)

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
            c[i], c[max] = c[max], c[i]

        v = R[i:,i].copy()
        norm = np.linalg.norm(v,ord=2)
        v[0] = v[0] - norm
        H = buildHouseholder(v,m,i)
        R = H@R
        Q = Q@H

        for k in range(i + 1, n):
            v = R[i+1:,k]
            c[k] = np.matmul(np.transpose(v),v)

    if(not foundRank):
        r = min([m,n])
    P = np.zeros([n,n])
    for i in range(n):
        P[p[i],i] = 1

    end = time.time()
    duration = end-start

    return [Q,R,P,r,duration]


def buildHouseholder(v,m,i):

    P = 2 * np.outer(v,np.transpose(v)) / (np.transpose(v)@v)
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
    Q = np.eye(m)
    p = np.array(range(n))

    if(k == 0):
        k = n
        forcedRank = False

    foundRank = False

    for j in range(n):
        v = R[:,j]
        c[j] = np.matmul(np.transpose(v),v)

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
            c[i], c[max] = c[max], c[i]
            p[i], p[max] = p[max], p[i]
        if(abs(maxNorm) < tol and not forcedRank):
            foundRank = True
            break
        if(i == m - 1):
            break
        r += 1

        

        for j in range(i + 1, m):
            G = buildGivens(j,i,R[i,i],R[j,i],m)
            R = np.matmul(G,R)
            Q = np.matmul(Q,np.transpose(G),Q)

        for k in range(i+1,n):
            v = R[i+1:,k]
            c[k] = np.matmul(np.transpose(v),v)
        

    if(not foundRank):
        r = min([m,n])

    P = np.zeros([n,n])
    for i in range(n):
        P[p[i],i] = 1

    end = time.time()

    duration = end - start

    return [Q,R,P,r,duration]
    


def buildGivens(i,j,a,b,m):

    r = np.hypot(a,b)
    c = a / r
    s = -b / r

    G = np.eye(m)
    G[i,i] = c
    G[j,j] = c
    G[i,j] = s
    G[j,i] = -s

    return G




# - only call driver if this file is run from terminal, prevents driver() from
#   being called if this file is imported into another file
if (__name__ == "__main__"):
    driver()