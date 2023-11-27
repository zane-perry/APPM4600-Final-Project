import numpy as np
import random
import scipy




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