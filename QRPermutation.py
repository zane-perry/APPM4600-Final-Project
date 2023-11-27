import numpy as np
import math
import random
import numpy.linalg as la
import scipy
import time


def driver():


    A = generate_matrix(4,4,4)

    [Ak, R11, R12, Q11] = QRrankKApproximation(A)

    
    

    




def permutedQR(A):

    start = time.time()
    Q = A.copy()

    [m,n] = Q.shape

    P = np.eye(n)

    for i in range(n):
        
        max = i
        maxNorm = np.linalg.norm(Q[:,i])
        for j in range(i + 1, n):
            norm = np.linalg.norm(Q[:,j])
            if(norm > maxNorm):
                max = j
                maxNorm = norm
        if(max != i):
            temp = Q[:,i].copy()
            Q[:,i] = Q[:,max]
            Q[:,max] = temp

            temp = P[:,i].copy()
            P[:,i] = P[:,max]
            P[:,max] = temp
        
        ui = Q[:,i]
        ui = ui / np.linalg.norm(ui)
        Q[:,i] = ui

        for k in range(i + 1, n):
            xk = Q[:,k]
            xk = xk - (np.inner(ui,xk) * ui)
            Q[:,k] = xk

    R = np.transpose(Q)@A@P

    end = time.time()
    duration = end - start
    
    return [Q,R,P,duration]


def QRrankKApproximation(A,k=0,tol=1.e-6):


    [m, n] = A.shape
    [Q,R,P,duration] = permutedQR(A)

    if(k==0):
        k = min([m,n])
        rii = np.diag(R)
        for i in range(k):
            if(abs(rii[i]) < tol):
                k = i
                break

    R11 = R[0:k,0:k]
    R12 = R[0:k, k:]
    Q11 = Q[:, 0:k]

    Ak = np.block([Q11@R11, Q11@R12])

    print('Time to create QR factorization:', duration)

    return [Ak, R11, R12, Q11]





def rand_sing(dim, rank, tol=1.e-6):
    # randomly generate rank singular values that are larger than the tol
    
    large_sing = np.zeros(rank)
    small_sing = np.zeros(dim - rank)
    
    for i in range(rank):
        large_sing[i] = random.uniform(10, 420)
        
    for j in range(dim - rank):
        small_sing[j] = random.uniform(0, tol)
        
    diag = np.append(large_sing, small_sing)

    np.random.shuffle(diag)

    
    diag_matrix = np.diag(diag)
    
    return diag_matrix


def generate_matrix(m,n,r):

    U = scipy.stats.ortho_group.rvs(m)
    V = scipy.stats.ortho_group.rvs(n)

    if(m < n):
        Sigma = rand_sing(m,r)
        block = np.zeros([m, n-m])
        Sigma = np.block([Sigma, block])
    elif(m > n):
        Sigma = rand_sing(n,r)
        block = np.zeros([m-n,n])
        Sigma = np.block([[Sigma],[block]])
    else:
        Sigma = rand_sing(m,r)

    return U@Sigma@V





driver()