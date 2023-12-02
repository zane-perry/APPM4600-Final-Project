import os

from helpers import recordAudioToDataVector, dataVectorToWavFile,\
    wavFileToDataVector

## convert .wav file to a representative data vector
##

currentDirectory = os.path.dirname(__file__)
fileToImport = "maedee_voice.wav"
fileLocation = os.path.join(currentDirectory, "Audio/", fileToImport)

sampleRate, dataVector = wavFileToDataVector(fileLocation)
print(dataVector)

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

