from helpers import computeSVD, permutedQR1

import numpy as np
import matplotlib.pyplot as plt


maxOrder = 200
orderList = np.array([i for i in range(0, maxOrder + 1)])
#print(orderList.shape)

trialsPerOrder = 50
minEleVal = -100
maxEleVal = 100

avgSVDTimeList = np.zeros([len(orderList)])
#print(avgSVDTimeList.shape)
avgQRTimeList = np.zeros([len(orderList)])
#print(avgQRTimeList.shape)

for order in orderList:
    avgSVDTime = 0
    avgQRTime = 0
    for trial in range(1, trialsPerOrder + 1):
        print("Order:", str(order), "Trial:", str(trial), end="\r")
        matrix = np.random.random_integers(minEleVal, maxEleVal, (order, order))
        (U, singularValues, V, svdDuration) = computeSVD(matrix)
        (Q, R, P, qrRank, qrDuration) = permutedQR1(matrix)
        avgSVDTime += svdDuration
        avgQRTime += qrDuration
    avgSVDTime /= trialsPerOrder
    avgQRTime /= trialsPerOrder
    
    avgSVDTimeList[order] = avgSVDTime
    avgQRTimeList[order] = avgQRTime

plt.plot(orderList, avgSVDTimeList)
plt.plot(orderList, avgQRTimeList)
plt.xlabel("Order of matrix")
plt.ylabel("Time (s)")
plt.legend(["Average time for SVD factorization",\
            "Average time for permuted QR factorization"])
plt.show()
