import numpy as np
import random
import scipy
import time
from generateDeficientMatrix import generate_matrix


def driver():


    A = generate_matrix(100,100,87)

    [Ak, R11, R12, Q11] = QRrankKApproximation1(A)

    #A = np.array([[16,15,14,13],[12,11,10,9],[8,7,6,5],[4,3,2,1]])
    #A = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
    #A = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
    #A = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]])

    [Ak, R11, R12, Q11] = QRrankKApproximation2(A)



def permutedQR1(A, k=0, tol=1.e-3):

    start = time.time()
    Q = A.copy()
    forcedRank = True
    [m,n] = Q.shape
    if(k == 0):
        k = n
        forcedRank = False

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
            v = Q[i:,k]
            c[k] = np.matmul(np.transpose(v),v)

    P = np.zeros([n,n])
    for i in range(n):
        P[p[i],i] = 1

    R = np.matmul(np.matmul(np.transpose(Q),A),P)

    end = time.time()
    duration = end - start
    
    return [Q,R,P,r,duration]



def QRGivens(A,k=0,tol=1.e-3):


    start = time.time()
    R = A.copy()
    [m,n] = R.shape
    c = np.zeros(n)
    Q = np.eye(m)
    p = np.array(range(n))


    for j in range(n):
        v = R[:,j]
        c[j] = np.matmul(np.transpose(v),v)

    for l in range(n):
        
        max = l
        maxNorm = c[l]
        for j in range(l + 1, n):
            norm = c[j]
            if(norm > maxNorm):
                max = j
                maxNorm = norm
        if(max != l):
            temp = R[:,l].copy()
            R[:,l] = R[:,max]
            R[:,max] = temp
            c[l], c[max] = c[max], c[l]
            p[l], p[max] = p[max], p[l]
        

        for j in range(l+1,n):
            c[j] = c[j] - (R[l,j]**2)

        for j in range(l):
            G = buildGivens(l,j,R[j,j],R[l,j],m)
            R = np.matmul(G,R)
            Q = np.matmul(Q,np.transpose(G),Q)

    P = np.zeros([n,n])
    for i in range(n):
        P[p[i],i] = 1

    end = time.time()

    duration = end - start

    return [Q,R,P,duration]
    

        




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



def QRrankKApproximation1(A,k=0,tol=1.e-3):


    if(k==0):
        [Q,R,P,k,duration] = permutedQR1(A)
    else:
        [Q,R,P,k,duration] = permutedQR1(A,k=k)

    R11 = R[0:k,0:k]
    R12 = R[0:k, k:]
    Q11 = Q[:, 0:k]

    Ak = np.block([np.matmul(Q11,R11), np.matmul(Q11,R12)])

    diff = np.linalg.norm(A - np.matmul(Ak,np.transpose(P)))

    print('Time to create QR factorization 1:', duration)
    print('Rank ', k, 'approximation')
    print('Error of ', diff)


    return [Ak, R11, R12, Q11]



def QRrankKApproximation2(A,k=0,tol=1.e-3):

    [m,n] = A.shape

    [Q,R,P,duration] = QRGivens(A,k=k,tol=tol)

    if(k==0):
        diagonals = np.diag(R)
        for i in range(min([m,n])):
            if(diagonals[i] < tol):
                break
            k += 1

    R11 = R[0:k,0:k]
    R12 = R[0:k, k:]
    Q11 = Q[:, 0:k]

    Ak = np.block([np.matmul(Q11,R11), np.matmul(Q11,R12)])

    diff = np.linalg.norm(A - np.matmul(Ak,np.transpose(P)))

    print('Time to create QR factorization 2:', duration)
    print('Rank ', k, 'approximation')
    print('Error of ', diff)


    return [Ak, R11, R12, Q11]



driver()