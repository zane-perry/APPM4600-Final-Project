from helpers import computeSVD, householderQR, generate_matrix

import numpy as np
import matplotlib.pyplot as plt


maxOrder = 200
orderList = np.array(range(3, maxOrder + 1))
#print(orderList.shape)

trialsPerOrder = 50

avgSVDTimeList = np.zeros([len(orderList)])
#print(avgSVDTimeList.shape)
avgQRTimeList = np.zeros([len(orderList)])
#print(avgQRTimeList.shape)

for i in range(len(orderList)):
    order = orderList[i]
    avgSVDTime = 0
    avgQRTime = 0
    for trial in range(1, trialsPerOrder + 1):
        print("Order:", str(order), "Trial:", str(trial), end="\r")
        #matrix = np.random.random_integers(minEleVal, maxEleVal, (order, order))
        matrix = generate_matrix(order,order,order)
        (U, singularValues, V, svdDuration) = computeSVD(matrix)
        (Q, R, P, qrRank, qrDuration) = householderQR(matrix)
        avgSVDTime += svdDuration
        avgQRTime += qrDuration
    avgSVDTime /= trialsPerOrder
    avgQRTime /= trialsPerOrder
    
    avgSVDTimeList[i] = avgSVDTime
    avgQRTimeList[i] = avgQRTime

plt.plot(orderList, avgSVDTimeList, label='SVD')
plt.plot(orderList, avgQRTimeList, label='Permuted QR')
plt.xlabel("Order of matrix")
plt.ylabel("Time (s)")
plt.title('Average Time for Matrix Factorization')
plt.legend()
plt.show()
