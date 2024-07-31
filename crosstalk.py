import numpy as np
import numpy.random as rand
from numpy import linalg as la
from scipy.io import wavfile

def rms_blocks(signal, n):
	"""compute RMS values over blocks of the signal"""
	b = len(signal) // n
	result = np.zeros((b,))
	scale = np.sqrt(1.0 / n)
	for i in range(b):
		result[i] = la.norm(signal[i*n:(i+1)*n]) * scale
	return result

def extend(left, sig, right):
	"""extends a signal to the left and right by duplicating the
	   border samples. The resulting ndarray will have a length
	   of left+len(sig)+right."""
	n = len(sig)
	tmp = np.zeros((left+n+right,))
	tmp[0:left] = sig[0]
	tmp[-right:] = sig[-1]
	tmp[left:left+n] = sig
	return tmp

def bidihysteresis(sig, levelon, leveloff):
	"""returns an ndarray of booleans depending on whether the
	   signal goes above/below the given thresholds. Samples above
	   levelon are set as True, samples below leveloff are set as
	   False. Inbetween, Truth values are extended to both sides
	   until the signal drops below leveloff."""
	n = len(sig)
	res = (sig >= levelon)
	dif = np.convolve(res*1.0,(1.0,-1.0),mode='valid')
	for i in np.where(dif > 0.5)[0]: # attacks
		j = i
		while j >= 0 and not res[j] and sig[j] >= leveloff:
			res[j] = True
			j -= 1
	for i in np.where(dif < -0.5)[0]: # releases
		j = i + 1
		while j < n and not res[j] and sig[j] >= leveloff:
			res[j] = True
			j += 1
	return res

def grow_envelope(sig):
	"""grows the envelope to both sides by two samples using max over a window of 5 samples"""
	tmp = extend(2,sig,2)
	return np.maximum(
		np.maximum(
			np.maximum(tmp[0:-4],tmp[1:-3]),
			np.maximum(tmp[2:-2],tmp[3:-1])
		),
		tmp[4:]
	)

def slope_attack(sig, max_delta):
	"""returns a modified sig by increasing values so that sig increases over time by at most max_delta"""
	sig = np.copy(sig)
	level = 0.0
	for i in reversed(range(len(sig))):
		level = max(sig[i], level - max_delta)
		sig[i] = level
	return sig

def slope_release(sig, max_delta):
	"""returns a modified sig by increasing values so that sig decreases over time not faster than max_delta"""
	sig = np.copy(sig)
	level = 0.0
	for i in range(len(sig)):
		level = max(sig[i], level - max_delta)
		sig[i] = level
	return sig

def med3filt(sig):
	"""median filter over three consecutive samples"""
	tmp = np.stack((sig[:-2],sig[1:-1],sig[2:]))
	tmp = np.median(tmp, axis=0)
	sig2 = np.copy(sig)
	sig2[1:-1] = tmp
	return sig2

def med5filt(sig):
	"""median filter over five consecutive samples"""
	tmp = np.stack((sig[:-4],sig[1:-3],sig[2:-2],sig[3:-1],sig[4:]))
	tmp = np.median(tmp, axis=0)
	sig2 = np.copy(sig)
	sig2[2:-2] = tmp
	return sig2

def smooth3filt(sig):
	"""simple 3-tap lowpass filter"""
	tmp = extend(1,sig,1)
	return np.convolve(tmp,(0.25,0.5,0.25),mode='valid')

def close_short_gaps(sig,maxgap):
	"""A "gap" of at most maxgap consecutive zeros between ones is "closed" by setting it to one."""
	delta = np.convolve(sig,(1.0,-1.0),mode='valid')
	sig = np.copy(sig)
	for i in np.where(delta > 0.5)[0]:
		j = i
		b = max(0, i - maxgap)
		while j >= b:
			if sig[j] > 0.5:
				sig[j+1:i+1] = 1.0
				j = b
			j -= 1
	return sig

def bsp2(n):
	"""returns the three segments (as tuple of ndarrays) of
	a second order base Spline"""
	x = (np.arange(n)+0.5) / n
	return (
		0.5*(x**2),
		0.75 - (x - 0.5)**2,
		0.5*((1.0-x)**2)
	)

def apply_envelope(sig, envelope, n):
	"""modulate signal by an interpolated envelope. The
	   envelope is assumed to have a lower sampling rate
	   by a factor of n. So, envelope[0] is for the first n
	   samples, envelope[1] is for the next n samples, etc.
	   But don't worry: This is smoothly interpolated
	   using a 2nd order Bspline (continuous 1st order
	   derivative)."""
	hullx = extend(1,envelope,1)
	sig2 = np.zeros(np.shape(sig))
	b = len(sig) // n
	bsp = bsp2(n)
	for i in range(b):
		curv = hullx[i  ]*bsp[2] + \
		       hullx[i+1]*bsp[1] + \
		       hullx[i+2]*bsp[0]
		sig2[i*n:(i+1)*n] = curv * sig[i*n:(i+1)*n]
	return sig2

def rms_subtract(x, y):
	"""assuming x = y + z where y and z are orthogonal,
	   compute the RMS of z given the RMS of x and y"""
	return np.sqrt(np.maximum(0.0, x**2 - y**2))

# model of audio recording
# ------------------------
# signal1 = speaker1 + filtered1(speaker2) * spk2_cross_gain + noise1
# signal2 = speaker2 + filtered2(speaker1) * spk1_cross_gain + noise2
#
# where "filtered1/2" changes spectral characteristics but not the RMS.
# We don't care about what it does exactly (high frequency dampening).
# We just care about RMS levels for VAD (voice activity detection).

class CrossTalk:
	def __init__(self, silence, spk1, spk2):
		"""Noise levels and cross-talk gains computed from
		   RMS levels for silence, speaker 1 talking and
		   speaker 2 talking where speaker 1 is expected to
		   be on the first channel and speaker 2 is expected
		   to be on the second channel. The parameters are all
		   pairs of RMS values for the two channels."""
		self.silence = silence
		expected_c1 = spk1[0] > spk2[0]
		expected_c2 = spk1[1] < spk2[1]
		if expected_c1 and expected_c2:
			pass  # nothing to do
		elif not expected_c1 and not expected_c2:
			spk1, spk2 = spk2, spk1  # swap speakers
		else:
			raise ValueError('speaker levels inconsistent?')
		self.spk1_cross_gain = \
			rms_subtract(spk1[1], silence[1]) / \
			rms_subtract(spk1[0], silence[0])
		self.spk2_cross_gain = \
			rms_subtract(spk2[0], silence[0]) / \
			rms_subtract(spk2[1], silence[1])

def noise_and_crosstalk_envelopes(rec1, rec2, ct):
	"""takes the envelopes of two recordings and some crosstalk metadata
	   to compute the envelopes of just the noise and cross talk (without
	   the signal) for later gating..."""
	sig1 = rms_subtract(rec1, ct.silence[0])
	sig2 = rms_subtract(rec2, ct.silence[1])

	# removing crosstalk in sig1/sig2...
	im = np.linalg.inv(
		((1.0, ct.spk2_cross_gain**2),
		 (ct.spk1_cross_gain**2, 1.0)) )
	sig1p = sig1**2
	sig2p = sig2**2
	sig1 = np.sqrt(np.maximum(im[0,0] * sig1p + im[0,1] * sig2p, 0.0))
	sig2 = np.sqrt(np.maximum(im[1,0] * sig1p + im[1,1] * sig2p, 0.0))

	threshold1 = np.hypot(ct.silence[0], sig2*ct.spk2_cross_gain)
	threshold2 = np.hypot(ct.silence[1], sig1*ct.spk1_cross_gain)
	return threshold1, threshold2

def quantize16(sig):
	"""quantize and clip a float vector to the 16 bit signed int range."""
	dither = np.convolve(rand.rand(len(sig) + 1), (1.0,-1.0), mode='valid')
	tmp = np.round(sig + dither)
	tmp[tmp >  32767] =  32767
	tmp[tmp < -32878] = -32768
	return tmp.astype('int16')

def smooth_gating(levels, levelon, leveloff, closing_max_gap = 12, attack_slope = 0.3, release_slope = 0.3):
	x = bidihysteresis(med3filt(levels), levelon, leveloff)
	x = grow_envelope(x * 1.0) # extend by 2 blocks on each side
	x = close_short_gaps(x, closing_max_gap)
	x = slope_attack(x, attack_slope)
	x = slope_release(x, release_slope)
	x = smooth3filt(x)
	return x

