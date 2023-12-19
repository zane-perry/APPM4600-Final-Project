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

whiteNoiseMaedataVector = addWhiteNoise(maedataVector, 0, 0.04, 10)
audioVectorToWavFile(whiteNoiseMaedataVector, maedeeSampleRate,\
                      "gaussian_0.04_new_maedee_voice")

testVector = np.random.rand(24000, 1)
testVector *= 20
# print(testVector.shape)
# print(testVector)
testVector = testVector.reshape(testVector.shape[0], 1)
# print(testVector)
# print(testVector.shape)

# cleanTestVector = removeWhiteNoiseSVD(testVector, 7500, 0.04, "LS", "AD",\
#                                       "SQRT(M)*ETA", 1, "BLOCKWISE", 0.03, 0, debug=True)


cleanMaedataVector = removeWhiteNoiseSVD(whiteNoiseMaedataVector,\
                                         maedeeSampleRate, 0.04, "MLS", "AD",\
                                            "SQRT(M)*ETA", 1, "BLOCKWISE",\
                                                windowDuration=0.03,\
                                                    overlapDuration=0.001,\
                                                        debug=True)



audioVectorToWavFile(cleanMaedataVector, maedeeSampleRate,\
                     "gaussian_0.04_MLS_AD_sqrtMeta_2_Blockwise_0")