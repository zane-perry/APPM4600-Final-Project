import sounddevice as sd
import scipy as sp

fs = 44100
seconds = 5

myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()  # Wait until recording is finished
sp.io.wavfile.write('output.wav', fs, myrecording)  # Save as WAV file 

