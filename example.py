import numpy as np
from scipy.io import wavfile
import crosstalk

if __name__ == '__main__':
	speaker1_recording_filename = 'speaker1_recorded.wav' # recording1 (microphone of speaker1)
	speaker2_recording_filename = 'speaker2_recorded.wav' # recording2 (microphone of speaker2)
	
	speaker1_processed_filename = 'speaker1_processed.wav' # processed version of recording1
	speaker2_processed_filename = 'speaker2_processed.wav' # processed version of recording1

	# block size for audio RMS analysis and gain/envelope processing in number of samples
	# (should be equivalent to about 10 milliseconds)
	chunk = 512  # appropriate for sampling rates around 48kHz

	# For three situations we can measure the RMS levels of both recordings:
	#
	# 1. when nobody speaks (just the noise floor)
	# 2. when only speaker1 speaks
	# 3. when only speaker2 speaks
	#
	# With a tool like Audacity we can easily measure RMS in dB. The below code
	# converts the levels from dB to 16-bit amplitudes assuming the recordings
	# are in 16-bit. Do NOT use -float('inf') as Db level for silence. silence
	# MUST be a positive number because otherwise speech activity thresholds
	# might be zero which will lead to a division by zero issue.
	#
	# RMS levels...                   rec1 ,  rec2
	silence = 32767*(10**(np.array(( -36.81, -36.81 ))/20.0))  # noise floor
	spk1    = 32767*(10**(np.array(( -12.62, -24.39 ))/20.0))  # when speaker 1 speaks
	spk2    = 32767*(10**(np.array(( -24.39, -12.62 ))/20.0))  # when speaker 2 speaks

	# extract cross talk parameters from RMS levels assuming that
	# . recording1 = clean_speaker1 + cg2to1 * clean_speaker2 + noise1
	# . recording2 = clean_speaker2 + cg1to2 * clean_speaker1 + noise2
	# . all audio signals (clean_speaker1, clean_speaker2, noise1, noise2) are
	#   orthogonal to each other which allows prediction of their RMS levels.
	ct = crosstalk.CrossTalk(silence, spk1, spk2)
	print('Based on the given RMS levels we estimate that')
	print('speaker1 in audible in channel2 with a gain factor of', ct.spk1_cross_gain)
	print('speaker2 in audible in channel1 with a gain factor of', ct.spk2_cross_gain)

	# Additional smooth gating parameters
	ct.spk1_cross_gain *= 1.1 # slightly overshoot estimated cross_gain
	ct.spk2_cross_gain *= 1.1 # slightly overshoot estimated cross_gain
	hysteresis_on  = 2.5      # RMS ratio between recorded level and dynamic speech activity threshold
	hysteresis_off = 2.0      # RMS ratio between recorded level and dynamic speech activity threshold
	closing_max_gap = 12      # max number of blocks for a gap (muted) after hysteresis that we want to UNmute
	attack_slope = 0.3        # positive gain changes are limited to 0.3 per block by "widening" the envelope
	release_slope = 0.3       # negative gain changes are limited to 0.3 per block by "widening" the envelope

	# load recordings
	print('loading recordings...')
	fs1,rec1 = wavfile.read(speaker1_recording_filename)
	fs2,rec2 = wavfile.read(speaker2_recording_filename)
	assert fs1 == fs2

	print('computing dynamic gating envelope functions...')

	# compute RMS levels over blocks of `chunk` samples.
	rec1_rms = crosstalk.rms_blocks(rec1, chunk) # compute RMS over blocks (decimating by factor `chunk`)
	rec2_rms = crosstalk.rms_blocks(rec2, chunk) # compute RMS over blocks (decimating by factor `chunk`)

	# "growing" envelopes helps with minor temporal misalignment of both recordings
	rec1_rms = crosstalk.grow_envelope(rec1_rms) # "grow" RMS envelope by two block in both directions
	rec2_rms = crosstalk.grow_envelope(rec2_rms) # "grow" RMS envelope by two block in both directions

	# compute expected noise levels INCLUDING cross talk for the gating effect...
	# we will use these for the "speach activity thresholds"
	noise1, noise2 = crosstalk.noise_and_crosstalk_envelopes(rec1_rms, rec2_rms, ct)

	# compute gain factors for "smooth gating" based on RMS ratios
	# of recorded/noise and hysteresis parameters
	gain1 = crosstalk.smooth_gating(rec1_rms / noise1, hysteresis_on, hysteresis_off,
		closing_max_gap=closing_max_gap, attack_slope=attack_slope, release_slope=release_slope)
	gain2 = crosstalk.smooth_gating(rec2_rms / noise2, hysteresis_on, hysteresis_off,
		closing_max_gap=closing_max_gap, attack_slope=attack_slope, release_slope=release_slope)

	# Interpolate gains as smooth envelope and apply on audio data
	print('applying envelopes to recordings...')
	p1 = crosstalk.apply_envelope(rec1, gain1, chunk)
	p2 = crosstalk.apply_envelope(rec2, gain2, chunk)

	# Save the processed audio files under a new name
	print('saving processed audio files...')
	wavfile.write(speaker1_processed_filename, fs1, crosstalk.quantize16(p1))
	wavfile.write(speaker2_processed_filename, fs2, crosstalk.quantize16(p2))

