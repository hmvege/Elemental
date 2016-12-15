from scipy.io.wavfile import write
from numpy import linspace,sin,pi,int16,arange,zeros,ones,exp
from matplotlib.pyplot import plot,show,axis, title
from sys import exit
import numpy as np

class Sound:
	def __init__(self, function, filename):
		self.function = function
		self.filename = filename

	def __call__(self, length, amplitude, Hz, channels, rate = 44100):
		function, filename = self.function, self.filename
		channels = arange(channels[0],channels[1]+1,dtype=int)
		function = [i(channels).reshape((len(channels),1)) for i in function]
		print function
		t = linspace(0, length, length*rate)
		self.rate, self.t, self.length, self.Hz = rate, t, length, Hz
		sinesum = 0
		env = self._envelope()
		for i in function:
			sinesum += env*sin(i*2*pi*t*Hz)
		data = np.sum(sinesum,axis=0)*amplitude
		self.tone = data.astype(int16)

		write('%s_%isec.wav' % (filename,length), rate, self.tone)
		print '%s_%isec.wav written.' % (filename,length)

	def plot(self):
		length, tone, filename, t = \
			self.length, self.tone, self.filename, self.t
		plot(t,tone)
		axis([0,length,15000,-15000])
		title('%s' % filename)
		show()

	def fourier(self):
		from pylab import fft
		t, tone = self.t, self.tone
		fftomega0 = abs(fft(tone))
		plot(t,fftomega0)
		show()

	def _sigmoid(self,a,b,n):
		# Sigmoid function used for creating the envelope
		half = (a+b)*10
		x = linspace(-half,half,n)
		return 1./(1+exp(-(x)))

	def _envelope(self):
		t = self.t
		# Envelope for 'smoothing' out the beginning and end of the signal
		N = self.rate*self.length
		envelope = ones(N)
		first_10 = int(0.05*N)
		last_10 = int(0.95*N)
		t10 = self.t[first_10]
		env = self._sigmoid(self.t[0],t10,first_10)
		envelope[:first_10] = env
		envelope[last_10:] = env[::-1]
		return envelope

	def play(self):
		import subprocess
		audio_file = "/Users/mathiasmamenvege/Programming/Programmering av lyder/%s_%isec.wav" % (self.filename,self.length)
		return_code = subprocess.call(["afplay", audio_file])

if __name__ == '__main__':
	# Lyman
	lyman_function = lambda n : 1-(1.0/n**2)
	# Lyman & Balmer
	balmer_function = lambda n : 1.0/4-(1.0/(n+1)**2)
	# Lyman & Balmer % Paschen
	paschen_function = lambda n : 1.0/9-(1.0/(n+2)**2)

	# lyman = Sound([lyman_function], 'Lyman_hydrogen')
	# lyman(length=60, amplitude=2000, Hz=440, channels=[2, 10], rate = 44100)
	# #lyman.fourier()
	# #lyman.play()
	# #lyman.plot()

	balmer = Sound([lyman_function, balmer_function], 'Balmer_hydrogen_test')
	balmer(length=60, amplitude=1000, Hz=880, channels=[2, 12], rate = 44100)
	#balmer.fourier()
	#balmer.play()
	#balmer.plot()

	# only_balmer = Sound([balmer_function], 'Balmer_hydrogen')
	# only_balmer(length=10, amplitude=1000, Hz=880, channels=[2, 12], rate = 96000)
	# only_balmer.play()
	# only_balmer.plot()


	# paschen = Sound([lyman_function, balmer_function,paschen_function], 'Paschen_hydrogen_test')
	# paschen(length=10, amplitude=1200, Hz=440, channels=[2, 12], rate = 44100)
	# # paschen.play()
	# # paschen.fourier()
	# # paschen.plot()