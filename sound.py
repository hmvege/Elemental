from numpy import pi, linspace, sin, asarray, zeros, ones, exp, sum, delete, fft
import matplotlib.pyplot as plt, re, sys
from scipy.io.wavfile import write


############################################################
# Verson 0.9
# 
# 
# 
# 
# Hente ut spektre fra
# http://physics.nist.gov/PhysRefData/ASD/lines_form.html
# Eksempelside:
# http://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra=C&low_wl=&upp_wn=&upp_wl=&low_wn=&unit=1&de=0&java_window=3&java_mult=&format=0&line_out=3&en_unit=0&output=0&page_size=15&show_obs_wl=1&order_out=0&max_low_enrg=&show_av=2&max_upp_enrg=&tsb_value=0&min_str=&A_out=0&max_str=&allowed_out=1&min_accur=&min_intens=&submit=Retrieve+Data
############################################################

class Sound:
	def __init__(self,file,filename):
		data = open(file)
		newdata = []
		for line in data:
			newline = re.findall('\d{2,15}.\d{1,10}',line)
			if newline != []:
				newdata.append(float(newline[0]))
		newdata2 = self._sveving(newdata)
		
		self.data, self.filename = asarray(newdata2[:int(len(newdata2))]), filename

	def __call__(self,length,Hz,amplitude,rate = 44100):
		if amplitude >= 0.1:
			sys.exit('Change amplitude! %g is way to high!' % amplitude)
		print self.data
		convertion_factor = 10**-3 # Converting to audible spectrum
		data, filename = 1./(self.data*convertion_factor), self.filename
		#data, filename = self.data*convertion_factor, self.filename
		print data
		N = length*rate
		t = linspace(0,length,N)
		tones = zeros((len(t),len(data)))
		self.t, self.N = t, N
		env = self._envelope()
		for i in range(len(data)):
			tones[:,i] = env*sin(1./data[i]*2*pi*t*Hz)

		tone = sum(amplitude*tones,axis=1)
		write('Elementer/%s_%ssec.wav' % (filename, length), rate, tone)
		print '%s_%ssec.wav written.' % (filename, length)
		self.tones, self.tone, self.length = tones, tone, length

	def _sveving(self,lines):
		newlines = lines
		min_eps = 5.0  		# Minimum avstand mellom toner
		max_eps = 10*10**4 	# 
		j = 0
		for i in range(0,len(lines)-1):
			if abs(newlines[i-j] - newlines[i+1-j]) < min_eps:
				newlines.pop((i-j+1))
				j += 1
			
		return [i for i in newlines if i < max_eps]

	def _sigmoid(self,a,b,n):
		# Sigmoid function used for creating the envelope
		half = (a+b)*10
		x = linspace(-half,half,n)
		return 1./(1+exp(-(x)))

	def _envelope(self):
		# Envelope for 'smoothing' out the beginning and end of the signal
		N = self.N
		envelope = ones(N)
		first_10 = int(0.1*N) 
		last_10 = int(0.9*N)
		t10 = self.t[first_10]
		env = self._sigmoid(self.t[0],t10,first_10)
		envelope[:first_10] = env
		envelope[last_10:] = env[::-1]
		return envelope

	def fourier(self):
		t, tone = self.t, self.tone
		fftomega0 = abs(fft.fft(tone))
		plt.plot(fftomega0[0:int(len(fftomega0)/2.)])
		plt.show()

	def play(self):
		import subprocess, os
		CURRENT_PATH = os.getcwd()
		audio_file = '%s/Elementer/%s_%ssec.wav' % (CURRENT_PATH, self.filename, self.length)
		print 'Playing %s_%ssec.wav' % (self.filename, self.length)
		return_code = subprocess.call(['afplay', audio_file])

	def plot(self):
		plt.plot(self.t, self.tones, color='b')
		plt.show()

# hydrogen = Sound('hydrogen_spectra.txt', 'H_1')
# hydrogen(length=10,Hz=440,amplitude=0.01,rate=44100)
# hydrogen.play()
# # hydrogen.fourier()
# # hydrogen.plot()

# helium = Sound('helium_spectra.txt', 'He_1')
# helium(length=10,Hz=440,amplitude=0.01,rate=44100)
# helium.play()
# #helium.plot()

# litium = Sound('litium_spectra.txt', 'Li')
# litium(length=10,Hz=440,amplitude=0.01)
# litium.play()
# #litium.plot()

# carbon = Sound('carbon_spectra.txt', 'C')
# carbon(length=60,Hz=440,amplitude=0.01)
# carbon.play()
# carbon.fourier()

# uranium = Sound('uranium_spectra.txt', 'U_test')
# uranium(length=5,Hz=440,amplitude=0.01)
# uranium.play()
# #uranium.plot()

# plutonium = Sound('plutonium_spectra.txt', 'Pu')
# plutonium(length=10,Hz=440,amplitude=0.01)
# plutonium.play()
# #plutonium.plot()