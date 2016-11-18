from struct import pack
from numpy import sin, pi

def sound(t, frequency, samplerate):
	data = 0
	for n in range(1,11):
		data += sin(2*pi*t)
	#return sin(2*pi*sin(10*t**2)*(frequency/float(samplerate)))
	return data*(frequency/float(samplerate))



def au_file(name='test.au', frequency=440, duration=1000, volume=0.5, file_size=8, samplerate=8000):

	filename = open(name, 'wb')
	# header needs size, encoding=2, sampling_rate=8000, channel=1
	filename.write(pack('>4s5L', '.snd'.encode("utf8"), 24, file_size*duration, 2, samplerate, 1))
	
	#function = 2 * pi * frequency/8000

	# write data
	for t in range(file_size * duration):
		# sine wave calculations
		#sin_seg = sin(seg * function)
		sin_segment = sound(t, frequency, samplerate)
		value = pack('b', int(volume * 127 * sin_segment))
		filename.write(value)
	filename.close()
	print("File %s written" % name)

# test the module ...
if __name__ == '__main__':
	freq = 3951.1
	au_file(name='sound%g.au' % freq, frequency=freq,
			duration=40000, volume=0.8, file_size=16, samplerate=48000)


	import subprocess
	audio_file = "/Users/mathiasmamenvege/Programming/Programmering av lyder/sound%g.au" % freq
	print "Playing sound%g.au" % freq
	return_code = subprocess.call(["afplay", audio_file])
