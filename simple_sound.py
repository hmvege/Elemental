from numpy import pi, linspace, sin, zeros, ones, exp
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

class SimpleSound():
	def __init__(self,filename):
		self.filename = filename

	def __call__(self,length=3,Hz=1000,amplitude=0.1,rate = 44100):
		N = length*rate
		t = linspace(0,length,N)

		self.length,self.Hz,self.amplitude,self.rate,self.N,self.t = (
			length,Hz,amplitude,rate,N,t)

		wave = self._envelope()*amplitude*sin(2*pi*t*Hz)
		write('lydfiler/%s_%sHz.wav' % (self.filename,Hz),rate, wave)
		print '%s_%sHz.wav written.' % (self.filename,Hz)

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
		print envelope
		return envelope

	def play(self):
		import subprocess
		audio_file = '/Users/mathiasmamenvege/Programming/Programmering av lyder/lydfiler/%s_%sHz.wav' % (self.filename,self.Hz)
		return_code = subprocess.call(['afplay', audio_file])

if __name__ == '__main__':
	my_sound = SimpleSound('testlyd')
	my_sound(3,440,0.1)
	my_sound.play()