import os
import scipy as sp
import numpy as np
import math

from helpers import recordAudioToDataVector, dataVectorToWavFile,\
    wavFileToDataVector, SVDrankKApproximation, average_adiag


## - get data vector of maedee_voice.wav and white_noise.wav, add them together, 
##   and create .wav file from it
##

currentDirectory = os.path.dirname(__file__)

fileToImport = "new_maedee_voice.wav"
fileLocation = os.path.join(currentDirectory, "Audio/", fileToImport)
(maedeeSampleRate, maedataVector) = wavFileToDataVector(fileLocation)

fileToImport = "new_white_noise.wav"
fileLocation = os.path.join(currentDirectory, "Audio/", fileToImport)
(whiteNoiseSampleRate, whiteNoiseDataVector) = wavFileToDataVector(fileLocation)

fileToImport = "new_combined.wav"
fileLocation = os.path.join(currentDirectory, "Audio/", fileToImport)
(combinedSampleRate, combinedDataVector) = wavFileToDataVector(fileLocation)
print("Combined sample rate:", str(combinedSampleRate))
print("Size of combined data vector:", str(combinedDataVector.shape))
print(combinedDataVector)

# combinedDataVector = maedataVector + whiteNoiseDataVector
# dataVectorToWavFile(combinedDataVector, maedeeSampleRate, "combined")

## calculations to split combined data vector into "windows", each with a 
## duration of 30 ms
##

duration = 5 # [s]

# (maedeeSampleRate [samples] / 1 [s]) * (duration [s])
totalSamples = maedeeSampleRate * duration # [samples]

# (duration [s]) / ( (30 [ms] / 1 [window]) * (1 [s] / 1000 [ms]) ) =
numWindows = math.ceil(duration / 0.03) # [windows]

# (totalSamples [samples]) / numWindows [windows]
samplesPerWindow = math.floor(totalSamples / numWindows) # [samples / window]

# (numWindows [windows]) * (samplesPerWindow [samples] / 1 [window])
numCoveredSamples = numWindows * samplesPerWindow # [samples]
numMissedSamples = totalSamples - numCoveredSamples # [samples]

print("Total samples:", str(totalSamples))
print("Number of 30 ms windows:", str(numWindows))
print("Number of samples per 30 ms window:", str(samplesPerWindow))
print("Samples covered:", numCoveredSamples)
print("Difference in samples:", numMissedSamples)

# # [0, totalSamples - 1], step: samplesPerWindow
# for i in range(0, totalSamples, samplesPerWindow):
#     windowVector = combinedDataVector[i : i + samplesPerWindow]
#     #print(windowVector)

windowStartIndex = 0
noiselessDataVector = np.zeros([totalSamples, 1])
#print("Size of noiseless vector:", str(noiselessDataVector.shape))
for i in range(0, numWindows):
    if i == numWindows - 1:
        windowVector = combinedDataVector[windowStartIndex:]
    else:
        windowVector = combinedDataVector[windowStartIndex : windowStartIndex +\
                                      samplesPerWindow]
    print(i, "/", numWindows - 1)
    #print("Length of window vector:", str(windowVector.shape))
    H = sp.linalg.hankel(windowVector)
    print("Rank of H:", str(np.linalg.matrix_rank(H)))
    k, P, Sigma, QT, Hhat, Pk, SigmaK, QTk, duration =\
        SVDrankKApproximation(H, k=200)
    s = Hhat[0]
    s = s.reshape(s.shape[0], 1)
    #print("Length of s vector:", str(s.shape))
    if i == numWindows - 1:
        noiselessDataVector[windowStartIndex:] = s
    else:
        noiselessDataVector[windowStartIndex : windowStartIndex +\
                            samplesPerWindow] = s
    
    #print("Current length of noiseless vector:", str(noiselessDataVector.shape))
    #print(noiselessDataVector)
    # print(i)
    # print(windowVector.shape)
    # print(windowVector)
    windowStartIndex += samplesPerWindow

print("Size of noiseless vector:", str(noiselessDataVector.shape))
print(noiselessDataVector)
dataVectorToWavFile(noiselessDataVector, maedeeSampleRate, "noiseless")





















## convert .wav file to a representative data vector
##

# currentDirectory = os.path.dirname(__file__)
# fileToImport = "maedee_voice.wav"
# fileLocation = os.path.join(currentDirectory, "Audio/", fileToImport)

# sampleRate, dataVector = wavFileToDataVector(fileLocation)
# print(dataVector)

## debugging 
##

# print("Inputted sample rate: ", sampleRate, "Hz")

# audioData = recordAudioToDataVector(sampleRate, duration)
# print("Size of recorded data vector: ", audioData.shape)
# print(audioData)

# dataVectorToWavFile(audioData, sampleRate, "output")

# detectedSampleRate, detectedAudioData = wavFileToDataVector("output.wav")
# print("Detected sample rate: ", detectedSampleRate, "Hz")
# print("Size of detected data vector: ", detectedAudioData.shape)
# print(detectedAudioData)

