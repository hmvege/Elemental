from numpy import pi, linspace, sin, int16, amax, asarray
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import re

class Sound:
	def __init__(self,file,filename):
		data = open(file)
		newdata = []
		for line in data:
			newline = re.findall('\d{2,15}',line)
			if newline != []:
				newdata.append(float(newline[0]))
		self.data, self.filename = asarray(newdata), filename

	def __call__(self):
		Pass
				
hydrogen = Sound('hydrogen_spectra.txt', 'H')

rate = 44100
length = 10 # Seconds
Hz = 440
amplitude = 1

# Configuring the length of the song as well as the rate
t = linspace(0,length,length*rate)

wave = amplitude*sin(2*pi*t*Hz)
#max_amplitude = amax(wave)
write('H_test.wav',rate, wave.astype(int16))

import subprocess
audio_file = "/Users/mathiasmamenvege/Programming/Programmering av lyder/H_test.wav"
return_code = subprocess.call(["afplay", audio_file])


# plt.plot(t,wave)
# plt.axis([0,length,max_amplitude,-max_amplitude])
# plt.show()