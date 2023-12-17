import os
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import math

from helpers import wavFileToAudioVector, audioVectorToWavFile, addWhiteNoise

# import new_maedee_voice.wav as the audio vector maedataVector

currentDirectory = os.path.dirname(__file__)

fileToImport = "new_maedee_voice.wav"
fileLocation = os.path.join(currentDirectory, "Audio/", fileToImport)
(maedeeSampleRate, maedataVector) = wavFileToAudioVector(fileLocation)

# create whiteNoiseMaedataVector, store as file

whiteNoiseMaedataVector = addWhiteNoise(maedataVector, 0, 0.03)
audioVectorToWavFile(whiteNoiseMaedataVector, maedeeSampleRate,\
                     "gaussian_new_maedee_voice")
