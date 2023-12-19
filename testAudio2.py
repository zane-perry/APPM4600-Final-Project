import os
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import math

from helpers import wavFileToAudioVector, audioVectorToWavFile, addWhiteNoise,\
    removeWhiteNoiseSVD

# import new_maedee_voice.wav as the audio vector maedataVector

currentDirectory = os.path.dirname(__file__)

fileToImport = "new_maedee_voice.wav"
fileLocation = os.path.join(currentDirectory, "Audio/", fileToImport)
(maedeeSampleRate, maedataVector) = wavFileToAudioVector(fileLocation)

# create whiteNoiseMaedataVector, store as file

whiteNoiseMaedataVector = addWhiteNoise(maedataVector, 0, 0.04)
audioVectorToWavFile(whiteNoiseMaedataVector, maedeeSampleRate,\
                      "gaussian_0.04_new_maedee_voice")

testVector = np.zeros(100)
# print(testVector.shape)
# print(testVector)
testVector = testVector.reshape(testVector.shape[0], 1)
# print(testVector)
# print(testVector.shape)

cleanTestVector = removeWhiteNoiseSVD(testVector, 10, 0.04, "tbd", "tbd",\
                                      "tbd", 1, "BLOCKWISE", 0.9, 0.4)


# cleanMaedataVector = removeWhiteNoiseSVD(whiteNoiseMaedataVector,\
#                                          maedeeSampleRate, 0.04, "MLS", "AD",\
#                                             "SQRT(M)*ETA", 0.1, "BLOCKWISE",\
#                                                 windowDuration=0.027,\
#                                                     overlapDuration=0.005,\
#                                                         debug=False)



# audioVectorToWavFile(cleanMaedataVector, maedeeSampleRate,\
#                      "gaussian_0.04_MLS_AD_sqrtMeta_0.1_Blockwise_0")