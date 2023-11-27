import numpy as np
import math
import random
import numpy.linalg as la
import scipy
import time


def driver():


    A = generate_matrix(10,10,8)

    [Ak, R11, R12, Q11] = QRrankKApproximation1(A)



def permutedQR1(A, k=0, tol=1.e-6):

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
        c[j] = np.linalg.norm(Q[:,j])

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
            print(maxNorm)
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
            xk = xk - (np.inner(ui,xk) * ui)
            Q[:,k] = xk
            c[k] = np.linalg.norm(Q[i:,k])

    P = np.zeros([n,n])
    for i in range(n):
        P[p[i],i] = 1

    R = np.transpose(Q)@A@P

    end = time.time()
    duration = end - start
    
    return [Q,R,P,r,duration]

def QRrankKApproximation1(A,k=0,tol=1.e-6):


    [m, n] = A.shape
    [Q,R,P,r,duration] = permutedQR1(A)

    if(k==0):
        [Q,R,P,k,duration] = permutedQR1(A)
    else:
        [Q,R,P,k,duration] = permutedQR1(A,k=k)

    R11 = R[0:k,0:k]
    R12 = R[0:k, k:]
    Q11 = Q[:, 0:k]

    Ak = np.block([Q11@R11, Q11@R12])

    diff = np.linalg.norm(A - Ak@np.transpose(P))

    print('Time to create QR factorization 1:', duration)
    print('Rank ', r, 'approximation')
    print('Error of ', diff)

    #print(A)
    #print(Q@R@np.transpose(P))
    print(np.diag(R))

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