import os
import scipy as sp

from helpers import recordAudioToDataVector, dataVectorToWavFile,\
    wavFileToDataVector, SVDrankKApproximation


## - get data vector of maedee_voice.wav and white_noise.wav, add them together, 
##   and create .wav file from it
##

currentDirectory = os.path.dirname(__file__)

fileToImport = "maedee_voice.wav"
fileLocation = os.path.join(currentDirectory, "Audio/", fileToImport)
(maedeeSampleRate, maedataVector) = wavFileToDataVector(fileLocation)

fileToImport = "white_noise.wav"
fileLocation = os.path.join(currentDirectory, "Audio/", fileToImport)
(whiteNoiseSampleRate, whiteNoiseDataVector) = wavFileToDataVector(fileLocation)

combinedDataVector = maedataVector + whiteNoiseDataVector
dataVectorToWavFile(combinedDataVector, maedeeSampleRate, "combined")

## (try to) remove the white noise from the combined data vector
##

# - split data vector into 30 millisecond windows
#   (maedeeSampleRate [samples] / 1 [s]) * (5 [s]) = 
#   5 * maedeeSampleRate [samples]
totalSamples = maedeeSampleRate * 5
# (5 [s]) / ( (30 [ms] / 1 [window]) * (1 [s] / 1000 [ms]) ) = 5/0.03 [windows]
numWindows = 5 / 0.03
# (totalSamples [samples]) / numWindows [windows] = [samples / window]
samplesPerWindow = totalSamples / numWindows
















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

