"""
	Definition of several array geometries and the AcousticScene class to store everything needed to simulate the
	trajectories and the DOA estimation results.
	Pytorch datasets for sound source signals and for AcousticScenes with random trajectories and with the LOCATA
	dataset recordings.

	File name: acousticTrackingDataset.py
	Author: David Diaz-Guerra
	Date creation: 03/2022
	Python Version: 3.8.1
	Pytorch Version: 1.8.1
"""

import numpy as np
import os
import re
import copy
from collections import namedtuple
from torch.utils.data import Dataset
import torch
import scipy
import scipy.io.wavfile
import soundfile
import pandas
import warnings
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import webrtcvad
from itertools import permutations

from utils import rms_angular_error_deg

import gpuRIR
#gpuRIR.activateLUT(False)
#gpuRIR.activateMixedPrecision(True)


# %% Util functions

def acoustic_power(s):
	""" Acoustic power of after removing the silences.
	"""
	w = 512  # Window size for silent detection
	o = 256  # Window step for silent detection

	# Window the input signal
	s = np.ascontiguousarray(s)
	sh = (s.size - w + 1, w)
	st = s.strides * 2
	S = np.lib.stride_tricks.as_strided(s, strides=st, shape=sh)[0::o]

	window_power = np.mean(S ** 2, axis=-1)
	th = 0.01 * window_power.max()  # Threshold for silent detection
	return np.mean(window_power[np.nonzero(window_power > th)])


def cart2sph(cart):
	xy2 = cart[:,0]**2 + cart[:,1]**2
	sph = np.zeros_like(cart)
	sph[:,0] = np.sqrt(xy2 + cart[:,2]**2)
	sph[:,1] = np.arctan2(np.sqrt(xy2), cart[:,2]) # Elevation angle defined from Z-axis down
	sph[:,2] = np.arctan2(cart[:,1], cart[:,0])
	return sph


# %% Util classes

class Parameter:
	""" Random parammeter class.
	You can indicate a constant value or a random range in its constructor and then
	get a value acording to that with getValue(). It works with both scalars and vectors.
	"""
	def __init__(self, *args):
		if len(args) == 1:
			self.random = False
			self.value = np.array(args[0])
			self.min_value = None
			self.max_value = None
		elif len(args) == 2:
			self. random = True
			self.min_value = np.array(args[0])
			self.max_value = np.array(args[1])
			self.value = None
		else: 
			raise Exception('Parammeter must be called with one (value) or two (min and max value) array_like parammeters')
	
	def getValue(self):
		if self.random:
			return self.min_value + np.random.random(self.min_value.shape) * (self.max_value - self.min_value)
		else:
			return self.value


# Named tuple with the characteristics of a microphone array and definitions of the LOCATA arrays:
ArraySetup = namedtuple('ArraySetup', 'arrayType, orV, mic_pos, mic_orV, mic_pattern')

dicit_array_setup = ArraySetup(arrayType='planar', 
    orV = np.array([0.0, 1.0, 0.0]),
    mic_pos = np.array((( 0.96, 0.00, 0.00),
						( 0.64, 0.00, 0.00),
						( 0.32, 0.00, 0.00),
						( 0.16, 0.00, 0.00),
						( 0.08, 0.00, 0.00),
						( 0.04, 0.00, 0.00),
						( 0.00, 0.00, 0.00),
						( 0.96, 0.00, 0.32),
						(-0.04, 0.00, 0.00),
						(-0.08, 0.00, 0.00),
						(-0.16, 0.00, 0.00),
						(-0.32, 0.00, 0.00),
						(-0.64, 0.00, 0.00),
						(-0.96, 0.00, 0.00),
						(-0.96, 0.00, 0.32))), 
    mic_orV = np.tile(np.array([[0.0, 1.0, 0.0]]), (15,1)),
    mic_pattern = 'omni'
)

dummy_array_setup = ArraySetup(arrayType='planar', 
    orV = np.array([0.0, 1.0, 0.0]),
    mic_pos = np.array(((-0.079,  0.000, 0.000),
						(-0.079, -0.009, 0.000),
						( 0.079,  0.000, 0.000),
						( 0.079, -0.009, 0.000))), 
    mic_orV = np.array(((-1.0, 0.0, 0.0),
						(-1.0, 0.0, 0.0),
						( 1.0, 0.0, 0.0),
						( 1.0, 0.0, 0.0))), 
    mic_pattern = 'omni'
)

benchmark2_array_setup = ArraySetup(arrayType='3D', 
    orV = np.array([0.0, 1.0, 0.0]),
    mic_pos = np.array(((-0.028,  0.030, -0.040),
						( 0.006,  0.057,  0.000),
						( 0.022,  0.022, -0.046),
						(-0.055, -0.024, -0.025),
						(-0.031,  0.023,  0.042),
						(-0.032,  0.011,  0.046),
						(-0.025, -0.003,  0.051),
						(-0.036, -0.027,  0.038),
						(-0.035, -0.043,  0.025),
						( 0.029, -0.048, -0.012),
						( 0.034, -0.030,  0.037),
						( 0.035,  0.025,  0.039))), 
    mic_orV = np.array(((-0.028,  0.030, -0.040),
						( 0.006,  0.057,  0.000),
						( 0.022,  0.022, -0.046),
						(-0.055, -0.024, -0.025),
						(-0.031,  0.023,  0.042),
						(-0.032,  0.011,  0.046),
						(-0.025, -0.003,  0.051),
						(-0.036, -0.027,  0.038),
						(-0.035, -0.043,  0.025),
						( 0.029, -0.048, -0.012),
						( 0.034, -0.030,  0.037),
						( 0.035,  0.025,  0.039))),
    mic_pattern = 'omni'
)

eigenmike_array_setup = ArraySetup(arrayType='3D', 
    orV = np.array([0.0, 1.0, 0.0]),
    mic_pos = np.array((( 0.000,  0.039,  0.015),
						(-0.022,  0.036,  0.000),
						( 0.000,  0.039, -0.015),
						( 0.022,  0.036,  0.000),
						( 0.000,  0.022,  0.036),
						(-0.024,  0.024,  0.024),
						(-0.039,  0.015,  0.000),
						(-0.024,  0.024,  0.024),
						( 0.000,  0.022, -0.036),
						( 0.024,  0.024, -0.024),
						( 0.039,  0.015,  0.000),
						( 0.024,  0.024,  0.024),
						(-0.015,  0.000,  0.039),
						(-0.036,  0.000,  0.022),
						(-0.036,  0.000, -0.022),
						(-0.015,  0.000, -0.039),
						( 0.000, -0.039,  0.015),
						( 0.022, -0.036,  0.000),
						( 0.000, -0.039, -0.015),
						(-0.022, -0.036,  0.000),
						( 0.000, -0.022,  0.036),
						( 0.024, -0.024,  0.024),
						( 0.039, -0.015,  0.000),
						( 0.024, -0.024, -0.024),
						( 0.000, -0.022, -0.036),
						(-0.024, -0.024, -0.024),
						(-0.039, -0.015,  0.000),
						(-0.024, -0.024,  0.024),
						( 0.015,  0.000,  0.039),
						( 0.036,  0.000,  0.022),
						( 0.036,  0.000, -0.022),
						( 0.015,  0.000, -0.039))), 
    mic_orV = np.array((( 0.000,  0.039,  0.015),
						(-0.022,  0.036,  0.000),
						( 0.000,  0.039, -0.015),
						( 0.022,  0.036,  0.000),
						( 0.000,  0.022,  0.036),
						(-0.024,  0.024,  0.024),
						(-0.039,  0.015,  0.000),
						(-0.024,  0.024,  0.024),
						( 0.000,  0.022, -0.036),
						( 0.024,  0.024, -0.024),
						( 0.039,  0.015,  0.000),
						( 0.024,  0.024,  0.024),
						(-0.015,  0.000,  0.039),
						(-0.036,  0.000,  0.022),
						(-0.036,  0.000, -0.022),
						(-0.015,  0.000, -0.039),
						( 0.000, -0.039,  0.015),
						( 0.022, -0.036,  0.000),
						( 0.000, -0.039, -0.015),
						(-0.022, -0.036,  0.000),
						( 0.000, -0.022,  0.036),
						( 0.024, -0.024,  0.024),
						( 0.039, -0.015,  0.000),
						( 0.024, -0.024, -0.024),
						( 0.000, -0.022, -0.036),
						(-0.024, -0.024, -0.024),
						(-0.039, -0.015,  0.000),
						(-0.024, -0.024,  0.024),
						( 0.015,  0.000,  0.039),
						( 0.036,  0.000,  0.022),
						( 0.036,  0.000, -0.022),
						( 0.015,  0.000, -0.039))),
    mic_pattern = 'omni'
)

miniDSP_array_setup = ArraySetup(arrayType='planar',
	orV = np.array([0.0, 0.0, 1.0]),
	mic_pos = np.array((( 0.0000,  0.0430, 0.000),
						( 0.0372,  0.0215, 0.000),
						( 0.0372, -0.0215, 0.000),
						( 0.0000, -0.0430, 0.000),
						(-0.0372, -0.0215, 0.000),
						(-0.0372,  0.0215, 0.000))),
	mic_orV = np.array(((0.0, 0.0, 1.0),
						(0.0, 0.0, 1.0),
						(0.0, 0.0, 1.0),
						(0.0, 0.0, 1.0),
						(0.0, 0.0, 1.0),
						(0.0, 0.0, 1.0))),
	mic_pattern = 'omni'
)


class AcousticScene:
	""" Acoustic scene class.
	It contains everything needed to simulate a moving sound source moving recorded
	with a microphone array in a reverberant room.
	It can also store the results from the DOA estimation.
	"""
	def __init__(self, room_sz, T60, beta, SNR, array_setup, mic_pos, source_signal, fs, traj_pts, timestamps,
				 trajectory, t, DOA, n_sources=1):
		assert ((isinstance(source_signal, list) or isinstance(source_signal, tuple)) and len(source_signal)==n_sources)
		assert ((isinstance(traj_pts, list) or isinstance(traj_pts, tuple)) and len(traj_pts)==n_sources)
		assert ((isinstance(trajectory, list) or isinstance(trajectory, tuple)) and len(trajectory)==n_sources)
		assert ((isinstance(DOA, list) or isinstance(DOA, tuple)) and len(DOA)==n_sources)
		self.room_sz = room_sz				# Room size
		self.T60 = T60						# Reverberation time of the simulated room
		self.beta = beta					# Reflection coefficients of the walls of the room (make sure it corresponds with T60)
		self.SNR = SNR						# Signal to (omnidirectional) Noise Ration to simulate
		self.array_setup = array_setup		# Named tuple with the characteristics of the array
		self.mic_pos = mic_pos				# Position of the center of the array
		self.n_sources = n_sources			# Number of sources
		self.source_signal = [np.float32(source_signal[i]) for i in range(len(source_signal))]  # Source signal
		self.fs = fs						# Sampling frequency of the source signal and the simulations
		self.traj_pts = traj_pts 			# Trajectory points to simulate
		self.timestamps = timestamps		# Time of each simulation (it does not need to correspond with the DOA estimations)
		self.trajectory = [np.float32(trajectory) for i in range(len(trajectory))]		# Continuous trajectory
		self.t = np.float32(t)							# Continuous time
		self.DOA = [np.float32(DOA[i]) for i in range(len(DOA))] 						# Continuous DOA

	def simulate(self, separated_sources_simulation=False):
		""" Get the array recording using gpuRIR to perform the acoustic simulations.
		"""
		if self.T60 == 0:
			Tdiff = 0.1
			Tmax = 0.1
			nb_img = [1,1,1]
		else:
			Tdiff = gpuRIR.att2t_SabineEstimator(12, self.T60) # Use ISM until the RIRs decay 12dB
			Tmax = gpuRIR.att2t_SabineEstimator(40, self.T60)  # Use diffuse model until the RIRs decay 40dB
			if self.T60 < 0.15: Tdiff = Tmax # Avoid issues with too short RIRs
			nb_img = gpuRIR.t2n( Tdiff, self.room_sz )

		nb_mics = len(self.mic_pos)
		nb_traj_pts = len(self.traj_pts[0])
		nb_gpu_calls = min(int(np.ceil( self.fs * Tdiff * nb_mics * nb_traj_pts * np.prod(nb_img) / 10e9 )), nb_traj_pts)
		traj_pts_batch = np.ceil( nb_traj_pts / nb_gpu_calls * np.arange(0, nb_gpu_calls+1) ).astype(int)

		if separated_sources_simulation:
			mic_signals = np.zeros((self.n_sources+1, len(self.t), nb_mics))
		else:
			mic_signals = np.zeros((len(self.t), nb_mics))
		ac_pow = np.zeros(self.n_sources)
		for n in range(self.n_sources):
			RIRs_list = [ gpuRIR.simulateRIR(self.room_sz, self.beta,
											 self.traj_pts[n][traj_pts_batch[0]:traj_pts_batch[1],:], self.mic_pos,
											 nb_img, Tmax, self.fs, Tdiff=Tdiff,
											 orV_rcv=self.array_setup.mic_orV, mic_pattern=self.array_setup.mic_pattern) ]
			for i in range(1,nb_gpu_calls):
				RIRs_list += [	gpuRIR.simulateRIR(self.room_sz, self.beta,
												   self.traj_pts[n][traj_pts_batch[i]:traj_pts_batch[i+1],:], self.mic_pos,
												   nb_img, Tmax, self.fs, Tdiff=Tdiff,
												   orV_rcv=self.array_setup.mic_orV, mic_pattern=self.array_setup.mic_pattern) ]
			RIRs = np.concatenate(RIRs_list, axis=0)
			mic_signals_sim = gpuRIR.simulateTrajectory(self.source_signal[n], RIRs, timestamps=self.timestamps, fs=self.fs)
			if separated_sources_simulation:
				mic_signals[n,...] = mic_signals_sim[0:len(self.t), :]
			else:
				mic_signals += mic_signals_sim[0:len(self.t),:]

			# Omnidirectional noise TODO: ¿Cómo defino la SNR si hay multiples fuentes?
			dp_RIRs = gpuRIR.simulateRIR(self.room_sz, self.beta, self.traj_pts[n], self.mic_pos, [1,1,1], 0.1, self.fs,
										orV_rcv=self.array_setup.mic_orV, mic_pattern=self.array_setup.mic_pattern)
			dp_signals = gpuRIR.simulateTrajectory(self.source_signal[n], dp_RIRs, timestamps=self.timestamps, fs=self.fs)
			ac_pow[n] = np.mean([acoustic_power(dp_signals[:,i]) for i in range(dp_signals.shape[1])])

		noise = np.sqrt(ac_pow.mean()/10**(self.SNR/10)) * np.random.standard_normal((len(self.t), nb_mics))
		if separated_sources_simulation:
			mic_signals[-1, ...] = noise
		else:
			mic_signals += noise

		# Apply the propagation delay to the VAD information if it exists
		if hasattr(self, 'source_vad'):
			self.vad = []
			for n in range(self.n_sources):
				vad = gpuRIR.simulateTrajectory(self.source_vad[n], dp_RIRs, timestamps=self.timestamps, fs=self.fs)
				self.vad.append( vad[0:len(self.t),:].mean(axis=1) > vad[0:len(self.t),:].max()*1e-3 )
		self.vad = np.logical_or.reduce(self.vad)

		return mic_signals

	def get_rmsae(self, frames_to_exclude=0, exclude_silences=False):
		""" Returns the Root Mean Square Angular Error (degrees) of the DOA estimation.
		The scene need to have the fields DOAw and DOAw_pred with the DOA groundtruth and the estimation.
		"""
		DOAw_pred = torch.from_numpy(np.stack(self.DOAw_pred))[:, frames_to_exclude:, :]
		DOAw = torch.from_numpy(np.stack(self.DOAw))[:, frames_to_exclude:, :]
		errors, pairings = [], []
		for pairing in permutations(range(DOAw.shape[0])):
			pairings.append(pairing)
			if not exclude_silences:
				errors.append(rms_angular_error_deg(DOAw_pred[pairing, ...].view(-1, 2),
													DOAw.view(-1, 2)))
			else:
				silences = self.vad.mean(axis=1) < 2 / 3
				errors.append(rms_angular_error_deg(DOAw_pred[pairing, np.invert(silences), ...].view(-1, 2),
													DOAw[:,np.invert(silences),...].view(-1, 2)))
		min_idx = np.argmin(errors)
		self.DOAw_pred = [self.DOAw_pred[pairings[min_idx][s]] for s in range(DOAw_pred.shape[0])]
		return errors[min_idx].item()

	# TODO: Repalantear si tiene sentido cuando hay multiples fuentes
	def findMapMaximum(self, exclude_silences=False):
		""" Generates the field DOAw_est_max with the DOA estimation using the SRP-PHAT maximums
		and returns its RMSAE (in degrees) if the field DOAw exists with the DOA groundtruth.
		The scene need to have the field maps with the SRP-PHAT map of each window.
		You can choose whether to include the silent frames into the RMSAE computation or not.
		"""
		max_flat_idx = self.maps.reshape((self.maps.shape[0], -1)).argmax(1)
		theta_max_idx, phi_max_idx = np.unravel_index(max_flat_idx, self.maps.shape[1:])

		# Index to angle (radians)
		if self.array_setup.arrayType == 'planar':
			theta = np.linspace(0, np.pi/2, self.maps.shape[1])
		else:
			theta= np.linspace(0, np.pi, self.maps.shape[1])
		phi = np.linspace(-np.pi, np.pi, self.maps.shape[2]+1)
		phi = phi[:-1]
		DOAw_srpMax = np.stack((theta[theta_max_idx], phi[phi_max_idx]), axis=-1)
		self.DOAw_srpMax = DOAw_srpMax

		if not exclude_silences:
			if hasattr(self, 'DOAw'):
				return rms_angular_error_deg(torch.from_numpy(self.DOAw_srpMax),
														torch.from_numpy(self.DOAw))
		else:
			silences = self.vad.mean(axis=1) < 2/3
			self.DOAw_srpMax[silences] = np.NaN
			if hasattr(self, 'DOAw'):
				return rms_angular_error_deg(torch.from_numpy(DOAw_srpMax[np.invert(silences), :]),
														 torch.from_numpy(self.DOAw[np.invert(silences), :]) )

	def findMapMaximumIco(self, ico_grid, exclude_silences=False):
		""" Generates the field DOAw_est_max with the DOA estimation using the SRP-PHAT maximums
		and returns its RMSAE (in degrees) if the field DOAw exists with the DOA groundtruth.
		The scene need to have the field maps with the SRP-PHAT map of each window.
		You can choose whether to include the silent frames into the RMSAE computation or not.
		"""
		max_flat_idx = self.maps.reshape((self.maps.shape[0], -1)).argmax(1)
		chart_max_idx, h_max_idx, w_max_idx = np.unravel_index(max_flat_idx, self.maps.shape[1:])
		max_cart_coor = ico_grid[chart_max_idx, h_max_idx, w_max_idx, :]
		self.DOAw_srpMax = (cart2sph(max_cart_coor)[:,1:],)

		if not exclude_silences:
			if hasattr(self, 'DOAw'):
				return rms_angular_error_deg(torch.from_numpy(self.DOAw_srpMax[0]),
														torch.from_numpy(self.DOAw[0]))
		else:
			silences = self.vad.mean(axis=1) < 2/3
			self.DOAw_srpMax[0][silences] = np.NaN
			if hasattr(self, 'DOAw'):
				return rms_angular_error_deg(torch.from_numpy(self.DOAw_srpMax[0][np.invert(silences), :]),
														 torch.from_numpy(self.DOAw[0][np.invert(silences), :]) )

	def plotScene(self, view='3D'):
		""" Plots the source trajectory and the microphones within the room
		"""
		assert view in ['3D', 'XYZ', 'XY', 'XZ', 'YZ']

		fig = plt.figure()

		if view == '3D' or view == 'XYZ':
			ax = Axes3D(fig)
			ax.set_xlim3d(0, self.room_sz[0])
			ax.set_ylim3d(0, self.room_sz[1])
			ax.set_zlim3d(0, self.room_sz[2])

			for n in range(len(self.traj_pts)):
				ax.scatter(self.traj_pts[n][:,0], self.traj_pts[n][:,1], self.traj_pts[n][:,2])
				ax.text(self.traj_pts[n][0,0], self.traj_pts[n][0,1], self.traj_pts[n][0,2], 'start')
			ax.scatter(self.mic_pos[:,0], self.mic_pos[:,1], self.mic_pos[:,2])

			ax.set_title('$T_{60}$' + ' = {:.3f}s, SNR = {:.1f}dB'.format(self.T60, self.SNR))
			ax.set_xlabel('x [m]')
			ax.set_ylabel('y [m]')
			ax.set_zlabel('z [m]')

		else:
			ax = fig.add_subplot(111)
			plt.gca().set_aspect('equal', adjustable='box')

			if view == 'XY':
				ax.set_xlim(0, self.room_sz[0])
				ax.set_ylim(0, self.room_sz[1])
				for n in range(len(self.traj_pts)):
					ax.scatter(self.traj_pts[n][:,0], self.traj_pts[n][:,1])
					ax.text(self.traj_pts[n][0,0], self.traj_pts[n][0,1], 'start')
				ax.scatter(self.mic_pos[:,0], self.mic_pos[:,1])
				ax.legend(['Source trajectory', 'Microphone array'])
				ax.set_xlabel('x [m]')
				ax.set_ylabel('y [m]')
			elif view == 'XZ':
				ax.set_xlim(0, self.room_sz[0])
				ax.set_ylim(0, self.room_sz[2])
				for n in range(len(self.traj_pts)):
					ax.scatter(self.traj_pts[n][:,0], self.traj_pts[n][:,2])
					ax.text(self.traj_pts[n][0,0], self.traj_pts[n][0,2], 'start')
				ax.scatter(self.mic_pos[:,0], self.mic_pos[:,2])
				ax.legend(['Source trajectory', 'Microphone array'])
				ax.set_xlabel('x [m]')
				ax.set_ylabel('z [m]')
			elif view == 'YZ':
				ax.set_xlim(0, self.room_sz[1])
				ax.set_ylim(0, self.room_sz[2])
				for n in range(len(self.traj_pts)):
					ax.scatter(self.traj_pts[n][:,1], self.traj_pts[n][:,2])
					ax.text(self.traj_pts[n][0,1], self.traj_pts[n][0,2], 'start')
				ax.scatter(self.mic_pos[:,1], self.mic_pos[:,2])
				ax.legend(['Source trajectory', 'Microphone array'])
				ax.set_xlabel('y [m]')
				ax.set_ylabel('z [m]')

		plt.show()

	def plotDOA(self):
		""" Plots the groundtruth DOA
		"""
		fig = plt.figure()
		ax = fig.add_subplot(111)
		for n in range(len(self.DOA)):
			ax.plot(self.t, self.DOA[n] * 180/np.pi)

		ax.legend(['Elevation', 'Azimuth'])
		ax.set_xlabel('time [s]')
		ax.set_ylabel('DOA [$^\circ$]')

		plt.show()

	def plotEstimation(self, legned_loc='best', title=None, file_name=None):
		""" Plots the DOA groundtruth and its estimation.
		The scene need to have the fields DOAw and DOAw_pred with the DOA groundtruth and the estimation.
		If the scene has the field DOAw_srpMax with the SRP-PHAT estimation, it also plots it.
		"""
		fig = plt.figure()
		gs = fig.add_gridspec(7, 1)
		ax = fig.add_subplot(gs[0,0])
		for n in range(len(self.source_signal)):
			ax.plot(self.t, self.source_signal[n])
		plt.xlim(self.tw[0], self.tw[-1])
		plt.tick_params(axis='both', which='both', bottom=False, labelbottom=False, left=False, labelleft=False)
		if title is not None: plt.title(title)

		ax = fig.add_subplot(gs[1:,0])
		if hasattr(self, 'DOAw_srpMax'):
			# plt.gca().set_prop_cycle(None)
			ax.plot(self.tw, self.DOAw_srpMax[n][:,0] * 180 / np.pi, '.', markersize=4, color='#99cbed')
			ax.plot(self.tw, self.DOAw_srpMax[n][:,1] * 180 / np.pi, '.', markersize=4, color='#ffcc9f')
		for n in range(len(self.DOAw)):
			ax.plot(self.tw, self.DOAw[n] * 180/np.pi)
		plt.gca().set_prop_cycle(None)
		for n in range(len(self.DOAw_pred)):
			ax.plot(self.tw, self.DOAw_pred[n] * 180/np.pi, '--')

		plt.legend(['Polar angle', 'Azimuth']*len(self.DOAw), loc=legned_loc)
		plt.xlabel('time [s]')
		plt.ylabel('DOA [$^\circ$]')

		silences = self.vad.mean(axis=1) < 2/3
		silences_idx = silences.nonzero()[0]
		start, end = [], []
		for i in silences_idx:
			if not i - 1 in silences_idx:
				start.append(i)
			if not i + 1 in silences_idx:
				end.append(i)
		for s, e in zip(start, end):
			plt.axvspan((s-0.5)*self.tw[1], (e+0.5)*self.tw[1], facecolor='0.5', alpha=0.5)

		plt.xlim(self.tw[0], self.tw[-1])
		if file_name is not None: fig.savefig(file_name)
		plt.show()

	def plotMap(self, w_idx):
		""" Plots the SRP-PHAT map of the window w_idx.
		If the scene has the fields DOAw, DOAw_pred, DOAw_srpMax it also plot them.
		"""
		maps = np.concatenate((self.maps, self.maps[..., 0, np.newaxis]), axis=-1)

		thetaMax = np.pi / 2 if self.array_setup.arrayType == 'planar' else np.pi
		theta = np.linspace(0, thetaMax, maps.shape[-2])
		phi = np.linspace(-np.pi, np.pi, maps.shape[-1])

		map = maps[w_idx, ...]
		DOA = [self.DOAw[n][w_idx, ...] for n in range(len(self.DOAw))] if hasattr(self, 'DOAw') else None
		DOA_pred = [self.DOAw_pred[n][w_idx, ...] for n in range(len(self.DOAw_pred))] if hasattr(self, 'DOAw_pred') else None
		DOA_srpMax = self.DOAw_srpMax[w_idx, ...] if hasattr(self, 'DOAw_srpMax') else None

		plot_srp_map(theta, phi, map, DOA, DOA_pred, DOA_srpMax)

	def animateScene(self, fps=10, file_name=None):
		""" Creates an animation with the SRP-PHAT maps of each window.
		The scene need to have the field maps with the SRP-PHAT map of each window.
		If the scene has the fields DOAw, DOAw_pred, DOAw_srpMax it also includes them.
		"""
		maps = np.concatenate((self.maps, self.maps[..., 0, np.newaxis]), axis=-1)
		thetaMax = np.pi/2 if self.array_setup=='planar' else np.pi
		theta = np.linspace(0, thetaMax, maps.shape[-2])
		phi = np.linspace(-np.pi, np.pi, maps.shape[-1])

		DOAw = self.DOAw if hasattr(self, 'DOAw') else None
		DOAw_pred = self.DOAw_pred if hasattr(self, 'DOAw_pred') else None
		DOAw_srpMax = self.DOAw_srpMax if hasattr(self, 'DOAw_srpMax') else None

		animate_trajectory(theta, phi, maps, fps, DOAw, DOAw_pred, DOAw_srpMax, file_name)


# %% Source signal Datasets

class LibriSpeechDataset(Dataset):
	""" Dataset with random LibriSpeech utterances.
	You need to indicate the path to the root of the LibriSpeech dataset in your file system
	and the length of the utterances in seconds.
	The dataset length is equal to the number of chapters in LibriSpeech (585 for train-clean-100 subset)
	but each time you ask for dataset[idx] you get a random segment from that chapter.
	It uses webrtcvad to clean the silences from the LibriSpeech utterances.
	"""

	def _exploreCorpus(self, path, file_extension, corpus_in_folders=True):
		directory_tree = {}
		if corpus_in_folders:
			for item in os.listdir(path):
				if os.path.isdir( os.path.join(path, item) ):
					directory_tree[item] = self._exploreCorpus( os.path.join(path, item), file_extension )
				elif item.split(".")[-1] == file_extension:
					directory_tree[ item.split(".")[0] ] = os.path.join(path, item)
		else:
			for item in os.listdir(path):
				sp_fn = re.split('[-.]', item)
				if len(sp_fn) == 4 and sp_fn[-1] == file_extension:
					if not sp_fn[0] in directory_tree: directory_tree[sp_fn[0]] = {}
					if not sp_fn[1] in directory_tree[sp_fn[0]]: directory_tree[sp_fn[0]][sp_fn[1]] = {}
					directory_tree[sp_fn[0]][sp_fn[1]][sp_fn[2]] = os.path.join(path, item)
		return directory_tree

	def _cleanSilences(self, s, aggressiveness, return_vad=False):
		self.vad.set_mode(aggressiveness)

		vad_out = np.zeros_like(s)
		vad_frame_len = int(10e-3 * self.fs)
		n_vad_frames = len(s) // vad_frame_len
		for frame_idx in range(n_vad_frames):
			frame = s[frame_idx * vad_frame_len: (frame_idx + 1) * vad_frame_len]
			frame_bytes = (frame * 32767).astype('int16').tobytes()
			vad_out[frame_idx*vad_frame_len: (frame_idx+1)*vad_frame_len] = self.vad.is_speech(frame_bytes, self.fs)
		s_clean = s * vad_out

		return (s_clean, vad_out) if return_vad else s_clean

	def __init__(self, path, T, size=None, return_vad=False, readers_range=None, file_extension='flac', corpus_in_folders=True):
		self.corpus = self._exploreCorpus(path, file_extension, corpus_in_folders)
		if readers_range is not None:
			for key in list(map(int, self.nChapters.keys())):
				if int(key) < readers_range[0] or int(key) > readers_range[1]:
					del self.corpus[key]

		self.nReaders = len(self.corpus)
		self.nChapters = {reader: len(self.corpus[reader]) for reader in self.corpus.keys()}
		self.nUtterances = {reader: {
				chapter: len(self.corpus[reader][chapter]) for chapter in self.corpus[reader].keys()
			} for reader in self.corpus.keys()}

		self.chapterList = []
		for chapters in list(self.corpus.values()):
			self.chapterList += list(chapters.values())

		self.fs = 16000
		self.T = T

		self.return_vad = return_vad
		self.vad = webrtcvad.Vad()

		self.sz = len(self.chapterList) if size is None else size

	def __len__(self):
		return self.sz

	def __getitem__(self, idx):
		if idx < 0: idx = len(self) + idx
		while idx >= len(self.chapterList): idx -= len(self.chapterList)
		chapter = self.chapterList[idx]

		# Get a random speech segment from the selected chapter
		s = np.array([])
		utt_paths = list(chapter.values())
		n = np.random.randint(0,len(chapter))
		while s.shape[0] < self.T * self.fs:
			utterance, fs = soundfile.read(utt_paths[n])
			assert fs == self.fs
			s = np.concatenate([s, utterance])
			n += 1
			if n >= len(chapter): n=0
		s = s[0: self.T * fs]
		s -= s.mean()

		# Clean silences, it starts with the highest aggressiveness of webrtcvad,
		# but it reduces it if it removes more than the 66% of the samples
		s_clean, vad_out = self._cleanSilences(s, 3, return_vad=True)
		if np.count_nonzero(s_clean) < len(s_clean) * 0.66:
			s_clean, vad_out = self._cleanSilences(s, 2, return_vad=True)
		if np.count_nonzero(s_clean) < len(s_clean) * 0.66:
			s_clean, vad_out = self._cleanSilences(s, 1, return_vad=True)

		return (s_clean, vad_out) if self.return_vad else s_clean


# %% Acoustic Scene Datasets

class RandomTrajectoriesDataset(Dataset):
	""" Dataset Acoustic Scenes with random trajectories.
	The length of the dataset is the length of the source signals dataset.
	When you access to an element you get both the simulated signals in the microphones and the AcousticScene object.
	"""
	def __init__(self, sourceDataset, room_sz, T60, abs_weights, array_setup, array_pos, SNR, nb_points,
				 n_sources=1, size=200, transforms=None, separated_sources_simulation=False, include_anechoic_simulation=False):
		"""
		sourceDataset: dataset with the source signals (such as LibriSpeechDataset)
		room_sz: Size of the rooms in meters
		T60: Reverberation time of the room in seconds
		abs_weights: Absorption coefficients rations of the walls
		array_setup: Named tuple with the characteristics of the array
		array_pos: Position of the center of the array as a fraction of the room size
		SNR: Signal to (omnidirectional) Noise Ration
		nb_points: Number of points to simulate along the trajectory
		n_sources: Number of sources with random trajectories in each scene
		size: Dataset size. Actually, the dataset is always infinite, it's only affects to the epoch sizes. [200]
		separated_sources_simulation: Do not summ the contribution of each source to the microphone signals. [False]
		transforms: Transform to perform to the simulated microphone signals and the Acoustic Scene
		include_anechoic_simulation: Include a third output with the anechoic simulation of the scenario [False]

		Any parameter (except from nb_points and transforms) can be Parameter object to make it random.
		"""
		self.sourceDataset = sourceDataset

		self.room_sz = room_sz if type(room_sz) is Parameter else Parameter(room_sz)
		self.T60 = T60 if type(T60) is Parameter else Parameter(T60)
		self.abs_weights = abs_weights if type(abs_weights) is Parameter else Parameter(abs_weights)

		assert np.count_nonzero(array_setup.orV) == 1, "array_setup.orV mus be parallel to an axis"
		self.array_setup = array_setup
		self.N = array_setup.mic_pos.shape[0]
		self.array_pos = array_pos if type(array_pos) is Parameter else Parameter(array_pos)

		self.SNR = SNR if type(SNR) is Parameter else Parameter(SNR)
		self.nb_points = nb_points
		self.fs = sourceDataset.fs

		self.n_sources = n_sources if type(n_sources) is Parameter else Parameter(n_sources)
		self.size = size
		self.separated_sources_simulation = separated_sources_simulation
		self.include_anechoic_simulation = include_anechoic_simulation

		self.transforms = transforms

	def __len__(self):
		return self.size #len(self.sourceDataset)

	def __getitem__(self, idx):
		acoustic_scene = self.getRandomScene(None)
		mic_signals = acoustic_scene.simulate(self.separated_sources_simulation)

		if self.include_anechoic_simulation:
			acoustic_scene_anechoic = copy.copy(acoustic_scene)
			acoustic_scene_anechoic.T60 = 0.0
			mic_signals_anechoic = acoustic_scene_anechoic.simulate(self.separated_sources_simulation)

		if self.transforms is not None:
			for t in self.transforms:
				mic_signals, acoustic_scene = t(mic_signals, acoustic_scene)
				if self.include_anechoic_simulation:
					mic_signals_anechoic = t(mic_signals_anechoic, None)[0]

		if self.include_anechoic_simulation:
			return mic_signals, acoustic_scene, mic_signals_anechoic
		else:
			return mic_signals, acoustic_scene

	def get_batch(self, idx1, idx2):
		mic_sig_batch = []
		acoustic_scene_batch = []
		mic_sig_anechoic_batch = []
		for idx in range(idx1, idx2):
			out_list = self[idx]
			mic_sig_batch.append(out_list[0])
			acoustic_scene_batch.append(out_list[1])
			if self.include_anechoic_simulation:
				mic_sig_anechoic_batch.append(out_list[2])

		if self.include_anechoic_simulation:
			return np.stack(mic_sig_batch), np.stack(acoustic_scene_batch), np.stack(mic_sig_anechoic_batch)
		else:
			return np.stack(mic_sig_batch), np.stack(acoustic_scene_batch)

	def getRandomScene(self, idx):
		n_sources = self.n_sources.getValue()

		# Source signal
		assert idx is None or n_sources==1
		if idx is not None:
			source_signal, vad = self.sourceDataset[idx]
			source_signal = (source_signal,)
			vad = (vad,)
		else:
			indexes = np.random.randint(0, len(self.sourceDataset), n_sources)
			sss_and_vads = [self.sourceDataset[indexes[n]] for n in range(n_sources)]
			sss_and_vads = list(zip(*sss_and_vads))
			source_signal = sss_and_vads[0]
			vad = sss_and_vads[1]


		# Room
		room_sz = self.room_sz.getValue()
		T60 = self.T60.getValue()
		abs_weights = self.abs_weights.getValue()
		beta = gpuRIR.beta_SabineEstimation(room_sz, T60, abs_weights)

		# Microphones
		array_pos = self.array_pos.getValue() * room_sz
		mic_pos = array_pos + self.array_setup.mic_pos

		# Trajectory points
		src_pos_min = np.array([0.0, 0.0, 0.0])
		src_pos_max = room_sz
		if self.array_setup.arrayType == 'planar':
			if np.sum(self.array_setup.orV) > 0:
				src_pos_min[np.nonzero(self.array_setup.orV)] = array_pos[np.nonzero(self.array_setup.orV)]
			else:
				src_pos_max[np.nonzero(self.array_setup.orV)] = array_pos[np.nonzero(self.array_setup.orV)]
		traj_pts = list()
		for i in range(n_sources):
			src_pos_ini = src_pos_min + np.random.random(3) * (src_pos_max - src_pos_min)
			src_pos_end = src_pos_min + np.random.random(3) * (src_pos_max - src_pos_min)

			Amax = np.min(np.stack((src_pos_ini - src_pos_min,
										  src_pos_max - src_pos_ini,
										  src_pos_end - src_pos_min,
										  src_pos_max - src_pos_end)),
									axis=0)

			A = np.random.random(3) * np.minimum(Amax, 1) 			# Oscilations with 1m as maximum in each axis
			w = 2*np.pi / self.nb_points * np.random.random(3) * 2  # Between 0 and 2 oscilations in each axis

			traj_pts_tmp = np.array([np.linspace(i,j,self.nb_points) for i,j in zip(src_pos_ini, src_pos_end)]).transpose()
			traj_pts_tmp += A * np.sin(w * np.arange(self.nb_points)[:, np.newaxis])
			traj_pts.append(traj_pts_tmp)

		# Interpolate trajectory points
		timestamps = np.arange(self.nb_points) * len(source_signal[0]) / self.fs / self.nb_points
		t = np.arange(len(source_signal[0]))/self.fs
		trajectory = [np.array([np.interp(t, timestamps, traj_pts[j][:,i]) for i in range(3)]).transpose() for j in range(n_sources)]

		acoustic_scene = AcousticScene(
			room_sz = room_sz,
			T60 = T60,
			beta = beta,
			SNR = self.SNR.getValue(),
			array_setup = self.array_setup,
			mic_pos = mic_pos,
			n_sources = n_sources,
			source_signal = source_signal,
			fs = self.fs,
			t = t,
			traj_pts = traj_pts,
			timestamps = timestamps,
			trajectory = trajectory,
			DOA = [cart2sph(trajectory[n] - array_pos) [:,1:3] for n in range(n_sources)]
		)
		acoustic_scene.source_vad = vad

		return acoustic_scene


class RandomTrajectoryDataset(RandomTrajectoriesDataset):
	""" Old style dataset for backward compatibility.
	"""

	def __init__(self, sourceDataset, room_sz, T60, abs_weights, array_setup, array_pos, SNR, nb_points, transforms=None):
		"""
		sourceDataset: dataset with the source signals (such as LibriSpeechDataset)
		room_sz: Size of the rooms in meters
		T60: Reverberation time of the room in seconds
		abs_weights: Absorption coefficients rations of the walls
		array_setup: Named tuple with the characteristics of the array
		array_pos: Position of the center of the array as a fraction of the room size
		SNR: Signal to (omnidirectional) Noise Ration
		nb_points: Number of points to simulate along the trajectory
		transforms: Transform to perform to the simulated microphone signals and the Acoustic Scene

		Any parameter (except from nb_points and transforms) can be Parameter object to make it random.
		"""
		super().__init__(sourceDataset, room_sz, T60, abs_weights, array_setup, array_pos, SNR, nb_points,
				 n_sources=1, size=len(sourceDataset), transforms=transforms)

	def __getitem__(self, idx):
		if idx < 0: idx = len(self) + idx

		acoustic_scene = self.getRandomScene(idx)
		mic_signals = acoustic_scene.simulate()

		if self.transforms is not None:
			for t in self.transforms:
				mic_signals, acoustic_scene = t(mic_signals, acoustic_scene)

		return mic_signals, acoustic_scene


class LocataDataset(Dataset):
	""" Dataset with the LOCATA dataset recordings and its corresponding Acoustic Scenes.
	When you access to an element you get both the simulated signals in the microphones and the AcousticScene object.
	"""
	def __init__(self, path, array, fs, tasks=(1,3,5), recording=None, dev=False, transforms = None):
		"""
		path: path to the root of the LOCATA dataset in your file system
		array: string with the desired array ('dummy', 'eigenmike', 'benchmark2' or 'dicit')
		fs: sampling frequency (you can use it to downsample the LOCATA recordings)
		tasks: LOCATA tasks to include in the dataset (only one-source tasks are supported)
		recording: recordings that you want to include in the dataset (only supported if you selected only one task)
		dev: True if the groundtruth source positions are available
		transforms: Transform to perform to the simulated microphone signals and the Acoustic Scene
		"""
		assert array in ('dummy', 'eigenmike', 'benchmark2', 'dicit'), 'Invalid array.'
		assert recording is None or len(tasks) == 1, 'Specific recordings can only be selected for dataset with only one task'
		# for task in tasks: assert task in (1,3,5), 'Invalid task ' + str(task) + '.'

		self.path = path
		self.dev = dev
		self.array = array
		self.tasks = tasks
		self.transforms = transforms
		self.fs = fs

		self.vad = webrtcvad.Vad()
		self.vad.set_mode(3)

		if array == 'dummy':
			self.array_setup = dummy_array_setup
		elif array == 'eigenmike':
			self.array_setup = eigenmike_array_setup
		elif array == 'benchmark2':
			self.array_setup = benchmark2_array_setup
		elif array == 'dicit':
			self.array_setup = dicit_array_setup

		self.directories = []
		for task in tasks:
			task_path = os.path.join(path, 'task' + str(task))
			for recording in os.listdir( task_path ):
				arrays = os.listdir( os.path.join(task_path, recording) )
				if array in arrays:
					self.directories.append( os.path.join(task_path, recording, array) )
		self.directories.sort()

	def __len__(self):
		return len(self.directories)

	def __getitem__(self, idx):
		directory = self.directories[idx]
		mic_signals, fs = soundfile.read( os.path.join(directory, 'audio_array_' + self.array + '.wav') )
		if fs > self.fs:
			mic_signals = scipy.signal.decimate(mic_signals, int(fs/self.fs), axis=0)
			new_fs = fs / int(fs/self.fs)
			if new_fs != self.fs: warnings.warn('The actual fs is {}Hz'.format(new_fs))
			self.fs = new_fs
		elif fs < self.fs:
			raise Exception('The sampling rate of the file ({}Hz) was lower than self.fs ({}Hz'.format(fs, self.fs))

		# Remove initial silence
		start = np.argmax(mic_signals[:,0] > mic_signals[:,0].max()*0.15)
		mic_signals = mic_signals[start:,:]
		t = (np.arange(len(mic_signals)) + start)/self.fs

		df = pandas.read_csv( os.path.join(directory, 'position_array_' + self.array + '.txt'), sep='\t' )
		array_pos = np.stack((df['x'].values, df['y'].values,df['z'].values), axis=-1)
		array_ref_vec = np.stack((df['ref_vec_x'].values, df['ref_vec_y'].values,df['ref_vec_z'].values), axis=-1)
		array_rotation = np.zeros((array_pos.shape[0],3,3))
		for i in range(3):
			for j in range(3):
				array_rotation[:,i,j] = df['rotation_' + str(i+1) + str(j+1)]

		df = pandas.read_csv( os.path.join(directory, 'required_time.txt'), sep='\t' )
		required_time = df['hour'].values*3600+df['minute'].values*60+df['second'].values
		timestamps = required_time - required_time[0]

		if self.dev:
			source_signals = []
			sources_pos = []
			trajectories = []
			for file in os.listdir( directory ):
				if file.startswith('audio_source') and file.endswith('.wav'):
					source_signal, fs_src = soundfile.read(os.path.join(directory, file))
					if fs > self.fs:
						source_signal = scipy.signal.decimate(source_signal, int(fs_src / self.fs), axis=0)
					source_signals.append(source_signal[start:start+len(t)])
				if file.startswith('position_source'):
					df = pandas.read_csv( os.path.join(directory, file), sep='\t' )
					source_pos = np.stack((df['x'].values, df['y'].values,df['z'].values), axis=-1)
					sources_pos.append( source_pos )
					trajectories.append( np.array([np.interp(t, timestamps, source_pos[:,i]) for i in range(3)]).transpose() )
			# sources_pos = np.stack(sources_pos)
			# trajectories = np.stack(trajectories)

			DOA = []
			for s in range(len(sources_pos)):
				source_pos_local = np.matmul( np.expand_dims(sources_pos[s] - array_pos, axis=1), array_rotation ).squeeze() # np.matmul( array_rotation, np.expand_dims(sources_pos[s,...] - array_pos, axis=-1) ).squeeze()
				DOA_pts = cart2sph(source_pos_local)[:,1:3]
				DOA_temp = np.array([np.interp(t, timestamps, DOA_pts[:,i]) for i in range(2)]).transpose()
				DOA_temp[DOA_temp[...,1]<-np.pi, 1] += 2*np.pi
				DOA.append(DOA_temp)
		else:
			sources_pos = None
			DOA = None
			source_signal = np.NaN * np.ones((len(mic_signals),1))

		acoustic_scene = AcousticScene(
			room_sz = np.NaN * np.ones((3,1)),
			T60 = np.NaN,
			beta = np.NaN * np.ones((6,1)),
			SNR = np.NaN,
			array_setup = self.array_setup,
			mic_pos = np.matmul( array_rotation[0,...], np.expand_dims(self.array_setup.mic_pos, axis=-1) ).squeeze() + array_pos[0,:], # self.array_setup.mic_pos + array_pos[0,:], # Not valid for moving arrays
			source_signal = source_signals,
			fs = self.fs,
			t = t - start/self.fs,
			traj_pts = sources_pos,
			timestamps = timestamps - start/self.fs,
			trajectory = trajectories,
			DOA = DOA
		)

		vad = np.zeros_like(source_signals[0])
		vad_frame_len = int(10e-3 * self.fs)
		n_vad_frames = len(source_signals[0]) // vad_frame_len
		for frame_idx in range(n_vad_frames):
			frame = source_signals[0][frame_idx * vad_frame_len: (frame_idx + 1) * vad_frame_len]
			frame_bytes = (frame * 32767).astype('int16').tobytes()
			vad[frame_idx*vad_frame_len: (frame_idx+1)*vad_frame_len] = self.vad.is_speech(frame_bytes, int(self.fs))
		acoustic_scene.vad = vad

		mic_signals.transpose()
		if self.transforms is not None:
			for t in self.transforms:
				mic_signals, acoustic_scene = t(mic_signals, acoustic_scene)

		return mic_signals, acoustic_scene

	def get_batch(self, idx1, idx2):
		mic_sig_batch = []
		acoustic_scene_batch = []
		for idx in range(idx1, idx2):
			mic_sig, acoustic_scene = self[idx]
			mic_sig_batch.append(mic_sig)
			acoustic_scene_batch.append(acoustic_scene)

		return np.stack(mic_sig_batch), np.stack(acoustic_scene_batch)


# %% Transform classes

class Windowing(object):
	""" Windowing transform.
	Create it indicating the window length (K), the step between windows and an optional
	window shape indicated as a vector of length K or as a Numpy window function.
	"""
	def __init__(self, K, step, window=None):
		self.K = K
		self.step = step
		if window is None:
			self.w = np.ones(K)
		elif callable(window):
			try: self.w = window(K)
			except: raise Exception('window must be a NumPy window function or a Numpy vector with length K')
		elif len(window) == K:
			self.w = window
		else:
			raise Exception('window must be a NumPy window function or a Numpy vector with length K')

	def __call__(self, x, acoustic_scene):
		if x is None:
			Xw = None
		else:
			assert x.ndim == 2 or x.ndim == 3
			if x.ndim == 3:
				separated_sources = True
				n_sources = x.shape[0]
				x = x.transpose((1,2,0))
				x = x.reshape((x.shape[0], -1))
			else: separated_sources = False

			N_mics = x.shape[1]
			L = x.shape[0]
			N_w = np.floor(L/self.step - self.K/self.step + 1).astype(int)

			if self.K > L:
				raise Exception('The window size can not be larger than the signal length ({})'.format(L))
			elif self.step > L:
				raise Exception('The window step can not be larger than the signal length ({})'.format(L))

			# Pad the signal
			x = np.append(x, np.zeros((N_w * self.step + self.K - L, N_mics)), axis=0)
			L = x.shape[0]

			# Window the signal
			shape_Xw = (N_w, self.K, N_mics)
			strides_Xw = [self.step*N_mics, N_mics, 1]
			strides_Xw = [strides_Xw[i] * x.itemsize for i in range(3)]
			Xw = np.lib.stride_tricks.as_strided(x, shape=shape_Xw, strides=strides_Xw)
			Xw = Xw.transpose((0,2,1)) * self.w
			if separated_sources:
				Xw = Xw.reshape((N_w, N_mics//n_sources, n_sources, self.K))
				Xw = Xw.transpose((2, 0, 1, 3))

		if acoustic_scene is not None:
			N_dims = acoustic_scene.DOA[0].shape[1]
			L = acoustic_scene.DOA[0].shape[0]
			N_w = np.floor(L/self.step - self.K/self.step + 1).astype(int)

			# Pad the DOA
			DOA = [None] * len(acoustic_scene.DOA)
			for n in range(len(acoustic_scene.DOA)):
				DOA[n] = np.append(acoustic_scene.DOA[n],
								np.tile(acoustic_scene.DOA[n][-1,:].reshape((1,2)), [N_w*self.step+self.K-L, 1]),
								axis=0) # Replicate the last known DOA
			L = DOA[0].shape[0]

			# Window the DOA
			shape_DOAw = (N_w, self.K, N_dims)
			strides_DOAw = [self.step*N_dims, N_dims, 1]
			strides_DOAw = [strides_DOAw[i] * DOA[0].itemsize for i in range(3)]
			DOAw = [None] * len(DOA)
			for n in range(len(DOA)):
				DOAw[n] = np.lib.stride_tricks.as_strided(DOA[n], shape=shape_DOAw, strides=strides_DOAw)
				DOAw[n] = np.ascontiguousarray(DOAw[n])
				for i in np.flatnonzero(np.abs(np.diff(DOAw[n][..., 1], axis=1)).max(axis=1) > np.pi):
					DOAw[n][i,  DOAw[n][i,:,1]<0, 1] += 2*np.pi # Avoid jumping from -pi to pi in a window
				DOAw[n] = np.mean(DOAw[n], axis=1)
				DOAw[n][DOAw[n][:,1]>np.pi, 1] -= 2*np.pi
			acoustic_scene.DOAw = DOAw

			# Window the VAD if it exists
			if hasattr(acoustic_scene, 'vad'):
				vad = acoustic_scene.vad[:, np.newaxis]
				vad = np.append(vad, np.zeros((L - vad.shape[0], 1)), axis=0)

				shape_vadw = (N_w, self.K, 1)
				strides_vadw = [self.step * 1, 1, 1]
				strides_vadw = [strides_vadw[i] * vad.itemsize for i in range(3)]

				acoustic_scene.vad = np.lib.stride_tricks.as_strided(vad, shape=shape_vadw, strides=strides_vadw)[...,0]

			# Timestamp for each window
			acoustic_scene.tw = np.arange(0, (L-self.K), self.step) / acoustic_scene.fs

		return Xw, acoustic_scene


class Extract_DOAw(object):
	""" Replace the AcousticScene object by just its windowed DOA
	"""
	def __call__(self, x, acoustic_scene):
		return x, acoustic_scene.DOAw


class ToFloat32(object):
	""" Convert to np.float32
	"""
	def __call__(self, x, DOAw):
		return x.astype(np.float32), DOAw.astype(np.float32)


# %% Representation functions

def plot_srp_map(theta, phi, srp_map, DOA_list=None, DOA_est_list=None, DOA_srpMax_list=None, colorbar=False, title=None):
	theta = theta * 180/np.pi
	phi = phi * 180/np.pi
	theta_step = theta[1] - theta[0]
	phi_step = phi[1] - phi[0]
	plt.imshow(srp_map, cmap='inferno', extent=(phi[0]-phi_step/2, phi[-1]+phi_step/2, theta[-1]+theta_step/2, theta[0]-theta_step/2))
	if colorbar: plt.colorbar()
	plt.xlabel('Azimuth [$^\circ$]')
	plt.ylabel('Elevation [$^\circ$]')

	if DOA_list is not None:
		for DOA in DOA_list:
			if DOA.ndim == 1: plt.scatter(DOA[1]*180/np.pi, DOA[0]*180/np.pi, c='r')
			elif DOA.ndim == 2: # For drawing the previous points in animations
				DOA_s = np.split(DOA, (np.abs(np.diff(DOA[:, 1])) > np.pi).nonzero()[0] + 1) # Split when jumping from -pi to pi
				[plt.plot(DOA_s[i][:, 1]*180/np.pi, DOA_s[i][:, 0]*180/np.pi, 'r') for i in range(len(DOA_s))]
				plt.scatter(DOA[-1,1]*180/np.pi, DOA[-1,0]*180 / np.pi, c='r')
	if DOA_srpMax_list is not None:
		for DOA_srpMax in DOA_srpMax_list:
			if DOA_srpMax.ndim == 1: plt.scatter(DOA_srpMax[1] *180/np.pi, DOA_srpMax[0]*180/np.pi, c='k')
			elif DOA_srpMax.ndim == 2:
				DOA_srpMax_s = np.split(DOA_srpMax, (np.abs(np.diff(DOA_srpMax[:, 1])) > np.pi).nonzero()[0] + 1) # Split when jumping from -pi to pi
				[plt.plot(DOA_srpMax_s[i][:, 1]*180 / np.pi, DOA_srpMax_s[i][:, 0]*180 / np.pi, 'k') for i in range(len(DOA_srpMax_s))]
				plt.scatter(DOA_srpMax[-1,1]*180 / np.pi, DOA_srpMax[-1,0]*180 / np.pi, c='k')
	if DOA_est_list is not None:
		for DOA_est in DOA_est_list:
			if DOA_est.ndim == 1: plt.scatter(DOA_est[1]*180/np.pi, DOA_est[0]*180/np.pi, c='b')
			elif DOA_est.ndim == 2:
				DOA_est_s = np.split(DOA_est, (np.abs(np.diff(DOA_est[:, 1])) > np.pi).nonzero()[0] + 1) # Split when jumping from -pi to pi
				[plt.plot(DOA_est_s[i][:, 1]*180 / np.pi, DOA_est_s[i][:, 0]*180 / np.pi, 'b') for i in range(len(DOA_est_s))]
				plt.scatter(DOA_est[-1,1]*180 / np.pi, DOA_est[-1,0]*180 / np.pi, c='b')

	plt.xlim(phi.min(), phi.max())
	plt.ylim(theta.max(), theta.min())
	plt.title(title)
	plt.show()


def animate_trajectory(theta, phi, srp_maps, fps, DOA=None, DOA_est=None, DOA_srpMax=None, file_name=None):
	fig = plt.figure()

	def animation_function(frame, theta, phi, srp_maps, DOA=None, DOA_est=None, DOA_srpMax=None):
		plt.clf()
		srp_map = srp_maps[frame,:,:]
		if DOA is not None: DOA = [DOA[n][:frame+1,:] for n in range(len(DOA))]
		if DOA_est is not None: DOA_est = [DOA_est[n][:frame+1,:] for n in range(len(DOA_est))]
		if DOA_srpMax is not None: DOA_srpMax = [DOA_srpMax[n][:frame+1,:] for n in range(len(DOA_srpMax))]
		plot_srp_map(theta, phi, srp_map, DOA, DOA_est, DOA_srpMax)

	anim = animation.FuncAnimation(fig, animation_function, frames=srp_maps.shape[0], fargs=(theta, phi, srp_maps, DOA, DOA_est, DOA_srpMax), interval=1e3/fps, repeat=False)
	# plt.show()
	# plt.close(fig)
	if file_name is not None: anim.save(file_name, fps=fps, writer='imagemagick')#, extra_args=['-vcodec', 'libx264'])

