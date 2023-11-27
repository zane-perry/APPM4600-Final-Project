import numpy as np
import math
import random
import numpy.linalg as la
import scipy


def driver():

    #A = np.array([[-2,-3],[5,7],[-2,-2],[-4,-1]])
    #A = np.array([[1.,-1.,0.],[2.,0.,0.],[2.,2.,1.]])
    #A = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])

    A = generate_matrix(5,3,2)

    

    [Q,R,P] = permutedQR(A)
    
    [Qs,Rs,Ps] = scipy.linalg.qr(A, pivoting=True)
    
    
    print('Original Matrix')
    print(A)
    print('Permuted QR Matrices:')
    print(Q)
    print(R)
    print('Scipy Routine:')
    print(Qs)
    print(Rs)
    

    #print(A@P)
    #print(Q@R)

    




def permutedQR(A):

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
    
    return [Q,R,P]








def rand_sing(dim, rank, tol=1.e-6):
    # randomly generate rank singular values that are larger than the tol
    
    large_sing = np.zeros(rank)
    small_sing = np.zeros(dim - rank)
    
    for i in range(rank):
        large_sing[i] = random.uniform(10, 420)
        
    for j in range(dim - rank):
        small_sing[j] = random.uniform(10**(-16), tol)
        
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