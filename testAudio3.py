import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from helpers import QRrankKApproximation, findSpeechPause





def wavFileToDataVector(fileName: str):
    '''
    Convert a mono-channel .wav audio file to a numpy array\n
    Inputs:\n
    \t  fileName: name of file to convert WITH extension\n
    Outputs:\n
    \t  (sampleRate, audioVector) where:\n
    \t\t  sampleRate: detected sampleRate of .wav file, measured in Hz 
    \t\t\t  (number of samples per second)\n
    \t\t  audioVector: (m x 1) numpy array
    '''

    sampleRate, audioVector = sp.io.wavfile.read(fileName)

    return (sampleRate, audioVector)



def dataVectorToWavFile(audioVector: np.array, sampleRate, fileName: str):
    '''
    Convert a data vector of an audio recording to a .wav file\n
    Inputs:\n
    \t  audioVector: (m x 1) numpy vector to convert\n
    \t  sampleRate: sample rate of audioVector, measuring in Hz
    \t\t  (number of samples per second)\n
    \t  fileName: name of file to create WITHOUT the extension
    Outputs:\n
    \t None
    '''

    newFileName = fileName + ".wav"
    sp.io.wavfile.write(newFileName, sampleRate, audioVector)

    return


def buildNoisySignal(sbar,mu=0,eta2=1,snr=10):

    N = len(sbar)
    e = np.random.normal(mu,eta2,size=N)

    resizeCoeff = np.linalg.norm(sbar) * np.exp(-snr / 20) / np.linalg.norm(e)

    e = e * resizeCoeff


    return e




def cleanSegment(s,eta2,n=30,m=211,N=240):

    col = s[:m]
    row = s[m-1:]

    H = sp.linalg.hankel(col,row)

    mCov = np.matmul(np.transpose(H),H)

    k = QRrankKApproximation(mCov,tol=m*eta2)[0]

    Hhat = QRrankKApproximation(H,k=k)[4]

    shat = np.zeros(N)

    HhatFlipped = np.fliplr(Hhat)

    offset = m - 1

    for i in range(N):
        shat[i] = np.mean(np.diag(HhatFlipped, i - offset))

    return shat







currentDirectory = os.path.dirname(__file__)

'''
fileToImport = "new_maedee_voice.wav"
fileLocation = os.path.join(currentDirectory, "Audio/", fileToImport)
(sampleRate, sbar) = wavFileToDataVector(fileLocation)

e = buildNoisySignal(sbar)

s = sbar + e

dataVectorToWavFile(s,sampleRate,'noisySignal')

'''
fileToImport = "noisySignal.wav"
fileLocation = os.path.join(currentDirectory, "Audio/", fileToImport)
(sampleRate, s) = wavFileToDataVector(fileLocation)



pause = findSpeechPause(s,sampleRate,0.03)

eta2 = np.var(pause)


clipDuration = s.shape[0]

cleanedSignal = np.array([])



for i in range (0,clipDuration,240):

    si = s[i:i+240]

    sbari = cleanSegment(si,eta2)

    np.append(cleanedSignal,sbari)




dataVectorToWavFile(cleanedSignal,sampleRate,'cleanedSignal')








