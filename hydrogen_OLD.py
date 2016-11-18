from scipy.io.wavfile import write
from numpy import linspace,sin,pi,int16,arange,zeros
from matplotlib.pyplot import plot,show,axis, title
from sys import exit
import numpy as np

class Sound:
	def __init__(self, function, filename):
		self.function = function
		self.filename = filename

	def __call__(self, length, amplitude, Hz, channels, rate = 44100):
		function = self.function
		
		# Backup1
		# channels = arange(channels[0],channels[1]+1,dtype=int)
		# function = function(channels).reshape((len(channels),1))
		# t = linspace(0, length, length*rate)
		# data = np.sum(sin(function*2*pi*t*Hz),axis=0)*amplitude

		channels = arange(channels[0],channels[1]+1,dtype=int)
		function = [i(channels).reshape((len(channels),1)) for i in function]
		t = linspace(0, length, length*rate)
		sinesum = 0
		for i in function:
			sinesum += sin(i*2*pi*t*Hz)
		data = np.sum(sinesum,axis=0)*amplitude

		# Backup2
		# data = np.sum([sin(function(n)*2*pi*t*Hz) for n in range(channels[0], channels[1]+1)],axis=0)*amplitude
		# data = np.sum([sin(eval(function)*2*pi*t*Hz) for n in range(channels[0], channels[1]+1)],axis=0)*amplitude		
		# for n in range(channels[0], channels[1]+1):
		# 	data += eval(function)
		# data = data*amplitude
		
		self.tone, self.rate, self.t, self.length, self.Hz =\
			data.astype(int16), rate, t, length, Hz

		write('%s.wav' % self.filename, rate, self.tone)
		print '%s.wav written.' % self.filename

	def plot(self):
		length, tone, filename, t = \
			self.length, self.tone, self.filename, self.t
		plot(t,tone)
		axis([0,length,15000,-15000])
		title('%s' % filename)
		show()

if __name__ == '__main__':
	# Lyman
	#lyman_function = 'sin((1-(1.0/n**2))*2*pi*t*Hz)'
	lyman_function = lambda n : 1-(1.0/n**2)
	# Lyman & Balmer
	#balmer_function = '(sin((1-(1.0/n**2))*2*pi*t*Hz) + sin((1.0/4-(1.0/(n+1)**2))*2*pi*t*Hz))'
	balmer_function = lambda n : 1.0/4-(1.0/(n+1)**2)
	# Lyman & Balmer % Paschen
	paschen_function = lambda n : 1.0/9-(1.0/(n+2)**2)

	# lyman = Sound([lyman_function], 'Lyman_hydrogen')
	# lyman(length=10, amplitude=2000, Hz=440, channels=[2, 10], rate = 96000)
	# lyman.plot()

	# balmer = Sound([lyman_function, balmer_function], 'Balmer_hydrogen')
	# balmer(length=10, amplitude=1000, Hz=880, channels=[2, 12], rate = 96000)
	# balmer.plot()

	paschen = Sound([lyman_function, balmer_function,paschen_function], 'Paschen_hydrogen')
	paschen(length=10, amplitude=1200, Hz=880, channels=[2, 12], rate = 96000)
	# paschen.plot()