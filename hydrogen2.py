from scipy.io.wavfile import write
from numpy import linspace,sin,pi,arange,zeros,ones,exp,sum
from matplotlib.pyplot import plot,show,axis, title
from sys import exit
from numpy import sin,linspace,ones


class Sound:
	def __init__(self, n1, filename):
		self.n1 = n1
		self.filename = filename

	def _rydeberg(self):
		n1 = self.n1
		n2 = linspace(n1+1,n1+11,11)
		R = 1.097e1 # [m^-1]
		return (R*(1./n1**2 - 1./n2**2))

	def __call__(self, length, amplitude, Hz, rate = 44100):
		if amplitude >= 0.1:
			exit('Change amplitude! %g is way to high!' % amplitude)

		N = length*rate
		t = linspace(0,length,N)
		self.N,self.t = N,t
		#sinesum = 0
		data = self._rydeberg()

		tones = zeros((len(t),len(data)))
		env = self._envelope()
		for i in range(len(data)):
			tones[:,i] = env*sin(data[i]*2*pi*t*Hz)

		tone = sum(amplitude*tones,axis=1)
		print tone
		self.tone, self.rate, self.length, self.Hz =\
			tone, rate, length, Hz
		
		write('%s.wav' % self.filename, rate, self.tone)
		print '%s.wav written.' % self.filename

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

	def plot(self):
		length, tone, filename, t = \
			self.length, self.tone, self.filename, self.t
		plot(t,tone)
		#axis([0,length,15000,-15000])
		title('%s' % filename)
		show()

	def fourier(self):
		from pylab import fft
		t, tone = self.t, self.tone
		fftomega0 = abs(fft(tone))
		plot(t,fftomega0)
		show()

	def play(self):
		import subprocess
		audio_file = "/Users/mathiasmamenvege/Programming/Programmering av lyder/%s.wav" % self.filename
		return_code = subprocess.call(["afplay", audio_file])

if __name__ == '__main__':
	# Lyman
	lyman_function = lambda n : 1-(1.0/n**2)
	# Lyman & Balmer
	balmer_function = lambda n : 1.0/4-(1.0/(n+1)**2)
	# Lyman & Balmer % Paschen
	paschen_function = lambda n : 1.0/9-(1.0/(n+2)**2)

	lyman = Sound(3,'Lyman_test1')
	lyman(length=60, amplitude=0.09, Hz=440, rate = 44100)
	#lyman.plot()
	lyman.play()

	# balmer = Sound([lyman_function, balmer_function], 'Balmer_hydrogen')
	# balmer(length=10, amplitude=0.09, Hz=880, rate = 44100)
	# balmer.play()
	# balmer.plot()

	# paschen = Sound([lyman_function, balmer_function,paschen_function], 'Paschen_hydrogen')
	# paschen(length=10, amplitude=0.09, Hz=440, rate = 44100)
	# paschen.play()
	# #paschen.fourier()
	# # paschen.plot()