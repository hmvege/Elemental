import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import urllib2
import multiprocessing
import argparse
from scipy.io.wavfile import write as wavfile_write

import time # for benchmarking purposes

"""
Program for converting atomic spectra to sound.
"""

# Periodic table
periodic_table = [('Hydrogen', '1', 'H'), ('Helium', '2', 'He'), ('Lithium', '3', 'Li'), ('Beryllium', '4', 'Be'), ('Boron', '5', 'B'),
			('Carbon', '6', 'C'), ('Nitrogen', '7', 'N'), ('Oxygen', '8', 'O'), ('Fluorine', '9', 'F'), ('Neon', '10', 'Ne'), 
			('Sodium', '11', 'Na'), ('Magnesium', '12', 'Mg'), ('Aluminium', '13', 'Al'), ('Silicon', '14', 'Si'), 
			('Phosphorus', '15', 'P'), ('Sulfur', '16', 'S'), ('Chlorine', '17', 'Cl'), ('Argon', '18', 'Ar'), ('Potassium', '19', 'K'), 
			('Calcium', '20', 'Ca'), ('Scandium', '21', 'Sc'), ('Titanium', '22', 'Ti'), ('Vanadium', '23', 'V'), ('Chromium', '24', 'Cr'), 
			('Manganese', '25', 'Mn'), ('Iron', '26', 'Fe'), ('Cobalt', '27', 'Co'), ('Nickel', '28', 'Ni'), ('Copper', '29', 'Cu'), 
			('Zinc', '30', 'Zn'), ('Gallium', '31', 'Ga'), ('Germanium', '32', 'Ge'), ('Arsenic', '33', 'As'), ('Selenium', '34', 'Se'), 
			('Bromine', '35', 'Br'), ('Krypton', '36', 'Kr'), ('Rubidium', '37', 'Rb'), ('Strontium', '38', 'Sr'), ('Yttrium', '39', 'Y'), 
			('Zirconium', '40', 'Zr'), ('Niobium', '41', 'Nb'), ('Molybdenum', '42', 'Mo'), ('Technetium', '43', 'Tc'), 
			('Ruthenium', '44', 'Ru'), ('Rhodium', '45', 'Rh'), ('Palladium', '46', 'Pd'), ('Silver', '47', 'Ag'), ('Cadmium', '48', 'Cd'), 
			('Indium', '49', 'In'), ('Tin', '50', 'Sn'), ('Antimony', '51', 'Sb'), ('Tellurium', '52', 'Te'), ('Iodine', '53', 'I'), 
			('Xenon', '54', 'Xe'), ('Caesium', '55', 'Cs'), ('Barium', '56', 'Ba'), ('Lanthanum', '57', 'La'), ('Cerium', '58', 'Ce'), 
			('Praseodymium', '59', 'Pr'), ('Neodymium', '60', 'Nd'), ('Promethium', '61', 'Pm'), ('Samarium', '62', 'Sm'), 
			('Europium', '63', 'Eu'), ('Gadolinium', '64', 'Gd'), ('Terbium', '65', 'Tb'), ('Dysprosium', '66', 'Dy'), 
			('Holmium', '67', 'Ho'), ('Erbium', '68', 'Er'), ('Thulium', '69', 'Tm'), ('Ytterbium', '70', 'Yb'), ('Lutetium', '71', 'Lu'), 
			('Hafnium', '72', 'Hf'), ('Tantalum', '73', 'Ta'), ('Tungsten', '74', 'W'), ('Rhenium', '75', 'Re'), ('Osmium', '76', 'Os'), 
			('Iridium', '77', 'Ir'), ('Platinum', '78', 'Pt'), ('Gold', '79', 'Au'), ('Mercury', '80', 'Hg'), ('Thallium', '81', 'Tl'), 
			('Lead', '82', 'Pb'), ('Bismuth', '83', 'Bi'), ('Polonium', '84', 'Po'), ('Astatine', '85', 'At'), ('Radon', '86', 'Rn'), 
			('Francium', '87', 'Fr'), ('Radium', '88', 'Ra'), ('Actinium', '89', 'Ac'), ('Thorium', '90', 'Th'), ('Protactinium', '91', 'Pa'), 
			('Uranium', '92', 'U'), ('Neptunium', '93', 'Np'), ('Plutonium', '94', 'Pu'), ('Americium', '95', 'Am'), ('Curium', '96', 'Cm'), 
			('Berkelium', '97', 'Bk'), ('Californium', '98', 'Cf'), ('Einsteinium', '99', 'Es'), ('Fermium', '100', 'Fm'), 
			('Mendelevium', '101', 'Md'), ('Nobelium', '102', 'No'), ('Lawrencium', '103', 'Lr'), ('Rutherfordium', '104', 'Rf'), 
			('Dubnium', '105', 'Db'), ('Seaborgium', '106', 'Sg'), ('Bohrium', '107', 'Bh'), ('Hassium', '108', 'Hs'), 
			('Meitnerium', '109', 'Mt'), ('Darmstadtium', '110', 'Ds'), ('Roentgenium', '111', 'Rg'), ('Copernicium', '112', 'Cn'), 
			('Nihonium', '113', 'Nh'), ('Flerovium', '114', 'Fl'), ('Moscovium', '115', 'Mc'), ('Livermorium', '116', 'Lv'), 
			('Tennessine', '117', 'Ts'), ('Oganesson', '118', 'Og')]

def element_search(search_element):
	"""
	Searching the periodic table for elements. Returns false if desired element is not found.
	"""
	for element in periodic_table:
		if search_element == element[-1]:
			return search_element
	return False

def element_downloader(element, local_file=None):
	"""
	Function to retrieve element wavelengths
	"""
	if local_file:
		if local_file.split(".")[0] == "html":
			nist_webpage = local_file
			html = open(nist_webpage, 'r')
		else:
			# Trying to get the file as an array
			return np.loadtxt(local_file)
	else:
		# sys.exit('Quiting in avoidance to have to download anything...')
		nist_webpage = 'http://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra=%s&low_wl=&upp_wn=&upp_wl=&low_wn=&unit=1&de=0&java_window=3&java_mult=&format=0&line_out=0&en_unit=0&output=0&page_size=15&show_obs_wl=1&order_out=0&max_low_enrg=&show_av=3&max_upp_enrg=&tsb_value=0&min_str=&A_out=0&max_str=&allowed_out=1&min_accur=&min_intens=&submit=Retrieve+Data' % element
		html = urllib2.urlopen(nist_webpage)

	# Strings to match for in html-file
	pre_string = '\<td class\=\"fix\"\>[\W]+' # The [\W]+ matches all whitespace characters leading up to the wavelength number
	post_string = '\<' # Cut-off at "<".

	# Setting up for for-loop. Optimizing for-loop by predefining dot
	spectra = []
	spectra_append = spectra.append

	regex_sequence = re.compile(r'%s([\d\W\&nspb]+)%s' % (pre_string, post_string))
	regex_blanks = re.compile(r'[\&nspb;]+')

	# Retrieving data
	for match in regex_sequence.finditer(html.read()):
		spectra_append(match.groups()[0])

	# Removing blanks, converting to float, converting to array, dividing by on in order to achieve correct units!
	spectra = map(lambda line : float(regex_blanks.sub("",line)), spectra)

	return spectra

def create_tones(input_values):
	"""
	External function to use when parallelizing.
	Returns tone spectra as a t-length array.
	Fully vectorized.
	"""
	t, spectra, envelope, Hz = input_values
	t_modified = 2*np.pi*Hz*t
	tone_matrix = spectra[:,np.newaxis] * t_modified[np.newaxis,:]
	tone = envelope*np.sum(np.sin(tone_matrix),axis=0)
	return tone

def rydeberg(series, lyman=False, balmer=False, Paschen=False):
	"""
	The Rydeberg formula for finding the spectra wavelength
	"""
	n1 = self.n1
	n2 = linspace(n1+1,n1+11,11)
	R = 1.097e1 # [m^-1]
	return (R*(1./n1**2 - 1./n2**2))

class ElementSound:
	def __init__(self, element, local_file=None, filename=None, parallel=False, num_processors=4):
		# Checking if we are to run in parallel
		if parallel:
			self.parallel = True
			self.num_processors = num_processors
		else:
			self.parallel = False

		# Creating filename
		if not filename:
			self.filename = "%s" % element
		else:
			self.filename = filename

		# Downloading or retrieving element spectra data
		if not local_file:
			self.spectra = element_downloader(element)
			print "Spectra retrieved from nist.org."
		else:
			self.spectra = element_downloader(element,local_file)
			print "Spectra retrieved from local file %s." % local_file

	def create_sound(self, length=10, Hz=440, amplitude=0.01, sampling_rate=44100, convertion_factor=100, wavelength_cutoff=2.5e-1):
		"""
		Function for creating soundfile from experimental element spectra.
		"""
		if amplitude >= 0.1:
			sys.exit("Change amplitude! %g is way to high!" % amplitude)

		if length < 0.5:
			sys.exit("Cannot create a sample of length less than 0.5 seconds.")

		filename = self.filename
		spectra = self.spectra

		# Creating time vector and getting length of datapoints
		N = int(length*sampling_rate)
		t = np.linspace(0,length,N)

		# Setting up spectra and converting to audible spectrum
		spectra = (1./np.asarray(spectra))*convertion_factor
		if wavelength_cutoff:
			spectra = np.asarray([i for i in spectra if i > wavelength_cutoff])
		
		spectra_length = len(spectra) 

		# Creating envelope
		envelope = self._envelope(N, sampling_rate, length, amplitude)

		# # Type 1 (OLD METHOD)
		# pre_time = time.clock()
		# tones = np.zeros((N, spectra_length))
		# for i in xrange(spectra_length):
		# 	tones[:,i] = envelope*np.sin(spectra[i]*2*np.pi*t*Hz)
		# tone = np.sum(tones,axis=1)
		# post_time = time.clock()
		# old_time = (post_time - pre_time)
		# print "Time used on creating sounds bytes(OLD MEHTOD): %s seconds" % old_time

		# # Type 2 (VECTORIZED)
		# pre_time = time.clock()
		# tone = create_tones([t, spectra, envelope, Hz])
		# post_time = time.clock()
		# vec_time = (post_time - pre_time)
		# print "Time used on creating sounds bytes(NEW METHOD - VECTORIZED): %s seconds" % vec_time

		# # Type 3 (PARALLELIZED & VECTORIZED)
		# pre_time = time.clock()
		# num_processors = self.num_processors
		# input_values = zip(	np.split(t,num_processors),
		# 					[spectra for i in xrange(num_processors)],
		# 					np.split(envelope,num_processors),
		# 					[Hz for i in xrange(num_processors)])
		# pool = multiprocessing.Pool(processes=num_processors)
		# results = pool.map(create_tones,input_values) #improve t by splitting into num_processors
		# tone = np.concatenate(results)
		# post_time = time.clock()
		# par_time = (post_time - pre_time)
		# print 'Time used on creating sounds bytes(NEW METHOD - VECTORIZED & PARALLELIZED): %s seconds' % par_time
		# print "OLD/VEC = %g \nOLD/PAR = %g \nVEC/PAR = %g" % (old_time/vec_time, old_time/par_time, vec_time / par_time)

		if self.parallel:
			num_processors = self.num_processors
			input_values = zip(	np.split(t,num_processors),
								[spectra for i in xrange(num_processors)],
								np.split(envelope,num_processors),
								[Hz for i in xrange(num_processors)])
			pool = multiprocessing.Pool(processes=num_processors)
			results = pool.map(create_tones,input_values) #improve t by splitting into num_processors
			tone = np.concatenate(results)
		else:
			tone = create_tones([t, spectra, envelope, Hz])

		# Writing to file
		wavfile_write('Elementer/%s_%dsec.wav' % (filename, int(length)), sampling_rate, tone)
		print '%s_%dsec.wav written.' % (filename, int(length))

		self.tone, self.length = tone, length

	def __call__(self):
		sys.exit('Exiting: Calling not implemented.')

	def remove_beat(self, eps=1e-2):
		"""
		Removes beat frequencies that cause a beat destructively.
		"""
		spectra = self.spectra
		self.spectra = [spectra[i] for i in xrange(0,len(spectra)-1) if abs(spectra[i] - spectra[i+1]) > eps]

	def _envelope(self, N, sampling_rate, length, amplitude):
		"""
		Envelope for 'smoothing' out the beginning and end of the signal. Also sets the amplitude
		"""
		envelope_time = 0.2 # seconds
		envelope_N = int(sampling_rate*envelope_time)
		envelope_array = np.ones(N)
		envelope = self._envelope_function(envelope_N)
		envelope_array[:envelope_N] *= envelope
		envelope_array[(N - envelope_N):] *= envelope[::-1]

		return amplitude * envelope_array

	def _envelope_function(self, envelope_N):
		# Cosine envelope
		x = np.linspace(0,np.pi,envelope_N)
		return (1 - np.cos(x))/2.

def main(args):
	parser = argparse.ArgumentParser(prog='ElementSound', description='Program for converting atomic spectra to the audible spectrum')

	# Prints program version if prompted
	parser.add_argument('--version', action='version', version='%(prog)s 0.9')

	# Main argument, must have this one
	parser.add_argument('element',						default=False, 	type=str,	nargs=1,	help='takes the type of element. E.g. He')

	# Possible choices
	parser.add_argument('-lf', 	'--local_file',			default=None,	type=str,				help='takes a .html file for an atomic spectra from nist.org')
	parser.add_argument('-fn', 	'--filename',			default=None,	type=str,				help='output filename')
	parser.add_argument('-p', 	'--parallel',			default=False,	action='store_const',	const=True, help='enables running in parallel')
	parser.add_argument('-n', 	'--num_processors',		default=4,		type=int,				help='number of processors, default=4')
	parser.add_argument('-ln', 	'--length',				default=10,		type=float,				help='length in seconds, default=10')
	parser.add_argument('-hz',	'--hertz',				default=440,	type=int,				help='frequency, default=440')
	parser.add_argument('-amp',	'--amplitude',			default=0.01,	type=float,				help='amplitude of track, default=0.01')
	parser.add_argument('-sr',	'--sampling_rate',		default=44100,	type=int,				help='sampling rate, default=44100')
	parser.add_argument('-cf',	'--convertion_factor',	default=100,	type=float,				help='factor to pitch-shift spectra by, default=100')
	parser.add_argument('-wlc',	'--wavelength_cutoff',	default=2.5e-1,	type=float,				help='inverse wavelength to cutoff lower tones, default=2.5e-1.')
	parser.add_argument('-bt',  '--beat_cutoff', 		default=1e-2, 	type=float, 			help='removes one wavelength if two wavelengths have |lambda-lambda_0| > beat_cutoff, default=1e-2')

	args = parser.parse_args()
	element = args.element[0]
	if not element_search(element):
		sys.exit('Element %s not found.' % element)

	Sound = ElementSound(element, args.local_file, args.filename, args.parallel, args.num_processors)
	Sound.remove_beat(args.beat_cutoff)
	Sound.create_sound(args.length, args.hertz, args.amplitude, args.sampling_rate, args.convertion_factor, args.wavelength_cutoff)

	# pre_time = time.clock()
	# test = ElementSound('H','H.html',parallel=True)
	# # test = ElementSound('He','He_training.html',parallel=True)
	# # test = ElementSound('Pu',parallel=True)
	# test.remove_beat()
	# test.create_sound(convertion_factor=100, Hz=880, wavelength_cutoff=1e-2)
	# post_time = time.clock()
	# print 'Time used on program: %s seconds' % (post_time - pre_time)

if __name__ == '__main__':
	# print """TO-DO LIST:
	# [x]	Rework beat
	# [x]	Rework envelope
	# [x]	Vectorize main sound generator
	# [x]	Parallelize
	# [x]	Now correctly pitch-shifts wavelengths 
	# [x]	Make it accessible from the command line
	# [x]	Download spectra files for all elements and store locally
	# []	Implement the Rydeberg formula as a spectrum source
	# []	Final optimizations
	# """
	pre_time = time.clock()
	main(sys.argv[1:])
	post_time = time.clock()
	print 'Time used on program: %s seconds' % (post_time - pre_time)
