import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.mlab as ml
import librosa as li

from helpers import QRrankKApproximation



def driver():

    n = 30
    m = 211
    N = m + n - 1

    k = 8

    sbar = buildPureSignal(N)
    e = buildNoisySignal(sbar)
    s = sbar + e


    shat = signalApproximate(s,n,m,N,k)

    #Noisy and pure signal
    plt.figure()
    plt.magnitude_spectrum(sbar,Fs=8000,scale='linear')
    plt.magnitude_spectrum(s,Fs=8000,scale='linear')
    plt.magnitude_spectrum(shat,Fs=8000,scale='linear')
    plt.legend(['Pure Signal', 'Noisy Signal', 'Signal Approximation'])

    plt.show()




def buildPureSignal(N):

    i = np.arange(1,N+1,1)
    sbar = np.sin(0.4*i) + 2 * np.sin(0.9*i) + 4* np.sin(1.7*i) + 3 * np.sin(2.6 * i)

    return sbar



def buildNoisySignal(sbar,mu=0,eta2=1,snr=0):

    N = len(sbar)
    e = np.random.normal(mu,eta2,size=N)

    resizeCoeff = np.linalg.norm(sbar) * np.exp(-snr / 20) / np.linalg.norm(e)

    e = e * resizeCoeff


    return e



def signalApproximate(s,n,m,N,k):
    col = s[:m]
    row = s[m-1:]

    Hbar = sp.linalg.hankel(col,row)

    Hhat = QRrankKApproximation(Hbar,k=k)[4]

    shat = np.zeros(N)

    HhatFlipped = np.fliplr(Hhat)

    offset = m - 1

    for i in range(N):
        shat[i] = np.mean(np.diag(HhatFlipped, i - offset))

    return shat




driver()