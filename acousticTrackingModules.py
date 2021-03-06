"""
	Pytorch functions and layers for DOA estimation.

	File name: acousticTrackingModules.py
	Author: David Diaz-Guerra
	Date creation: 03/2022
	Python Version: 3.8.1
	Pytorch Version: 1.8.1
"""
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import icoCNN


# %% Neural Network layers

class CausConv3d(nn.Module):
	""" Causal 3D Convolution for SRP-PHAT maps sequences
	"""
	def __init__(self, in_channels, out_channels, kernel_size):
		super(CausConv3d, self).__init__()
		self.pad = kernel_size[0] - 1
		self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=(self.pad, 0, 0))

	def forward(self, x):
		return self.conv(x)[:, :, :-self.pad, :, :]


class CausConv2d(nn.Module):
	""" Causal 2D Convolution for spectrograms and GCCs sequences
	"""
	def __init__(self, in_channels, out_channels, kernel_size):
		super(CausConv2d, self).__init__()
		self.pad = kernel_size[0] - 1
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(self.pad, 0))

	def forward(self, x):
		return self.conv(x)[:, :, :-self.pad, :]


class CausConv1d(nn.Module):
	""" Causal 1D Convolution
	"""
	def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
		super(CausConv1d, self).__init__()
		self.pad = (kernel_size - 1) * dilation
		self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.pad, dilation=dilation)
		
	def forward(self, x):
		return self.conv(x)[:, :, :-self.pad]


class SoftArgMax(nn.Module):
	def __init__(self, input_shape, indexes=None, include_exp=False):
		super().__init__()
		self.input_shape = input_shape
		self.include_exp = include_exp
		self.n_dim_in = len(input_shape)

		if indexes is not None:
			self.n_dim_out = indexes.shape[0]
			self.register_buffer('indexes', indexes, persistent=False)
		else:
			self.n_dim_out = self.n_dim_in
			self.register_buffer('indexes', torch.empty((self.n_dim_out, ) + input_shape), persistent=False)
			for i in range(self.n_dim_in):
				broadcaster_shape = [1] * self.n_dim_in
				broadcaster_shape[i] = self.input_shape[i]
				self.indexes[i,...] = torch.arange(input_shape[i]).reshape(broadcaster_shape)

	def forward(self, x, return_map=False):
		assert x.shape[-self.n_dim_in:] == self.input_shape
		batch_shape = x.shape[:-self.n_dim_in]

		if self.include_exp: x = torch.exp(x - x.max())
		x = x / (x.sum(list(range(x.ndim-self.n_dim_in, x.ndim)), keepdim=True) + 1e-12)

		y = torch.empty(batch_shape + (self.n_dim_out,), device=x.device)
		for i in range(self.n_dim_out):
			y[...,i] = torch.sum(x * self.indexes[i,...], list(range(x.ndim-self.n_dim_in, x.ndim)))

		return (y, x) if return_map else y

	def cuda(self, device=None):
		for i in range(len(self.indexes)):
			self.indexes[i].cuda(device)

	def cpu(self):
		for i in range(len(self.indexes)):
			self.indexes[i].cpu()


# %% Signal processing and DOA estimation layers

class GCC(nn.Module):
	""" Compute the Generalized Cross Correlation of the inputs.
	In the constructor of the layer, you need to indicate the number of signals (N) and the window length (K).
	You can use tau_max to output only the central part of the GCCs and transform='PHAT' to use the PHAT transform.
	"""
	def __init__(self, N, K, tau_max=None, transform=None):
		assert transform is None or transform == 'PHAT', 'Only the \'PHAT\' transform is implemented'
		assert tau_max is None or tau_max <= K//2
		super(GCC, self).__init__()
		
		self.K = K
		self.N = N
		self.tau_max = tau_max if tau_max is not None else K//2
		self.transform = transform
	
	def forward(self, x):
		x_fft = torch.fft.rfft(x)
		if self.transform == 'PHAT':
			x_fft /= (x_fft.abs() + 1e-12)
		
		gcc = torch.empty(list(x_fft.shape[0:-2]) + [self.N, self.N, 2*self.tau_max+1], device=x.device)
		for n in range(self.N):
			gcc_fft_batch = x_fft[...,n,:].unsqueeze(-2) * x_fft.conj()
			gcc_batch = torch.fft.irfft(gcc_fft_batch, n=self.K)
			gcc[..., n, :, 0:self.tau_max+1] = gcc_batch[..., 0:self.tau_max+1]
			gcc[..., n, :, -self.tau_max:] = gcc_batch[..., -self.tau_max:]
			
		return gcc


class SRP_map(nn.Module):
	""" Compute the SRP-PHAT maps from the GCCs taken as input.
	In the constructor of the layer, you need to indicate the number of signals (N) and the window length (K), the
	desired resolution of the maps (resTheta and resPhi), the microphone positions relative to the center of the
	array (rn) and the sampling frequency (fs).
	With normalize=True (default) each map is normalized to ethe range [-1,1] approximately
	"""
	def __init__(self, N, K, resTheta, resPhi, rn, fs, c=343.0, normalize=True, avoid_negatives=False, thetaMax=np.pi/2):
		super(SRP_map, self).__init__()
		
		self.N = N
		self.K = K
		self.resTheta = resTheta
		self.resPhi = resPhi
		self.fs = float(fs)
		self.normalize = normalize
		self.avoid_negatives = avoid_negatives
		
		self.cross_idx = np.stack([np.kron(np.arange(N, dtype='int16'), np.ones((N), dtype='int16')), 
							 np.kron(np.ones((N), dtype='int16'), np.arange(N, dtype='int16'))])
		
		self.theta = np.linspace(0, thetaMax, resTheta)
		self.phi = np.linspace(-np.pi, np.pi, resPhi+1)
		self.phi = self.phi[0:-1]
		
		self.IMTDF = np.empty((resTheta, resPhi, self.N, self.N)) # Time differences, floats
		for k in range(self.N):
			for l in range(self.N):
				r = np.stack([np.outer(np.sin(self.theta), np.cos(self.phi)), np.outer(np.sin(self.theta), np.sin(self.phi)), np.tile(np.cos(self.theta), [resPhi, 1]).transpose()], axis=2)
				self.IMTDF[:,:,k,l] = np.dot( r, rn[l,:]-rn[k,:] ) / c

		tau = np.concatenate([range(0, K//2+1), range(-K//2+1, 0)])/float(fs) # Valid discrete values
		self.tau0 = np.zeros_like(self.IMTDF, dtype=np.int)
		for k in range(self.N):
			for l in range(self.N):
				for i in range(resTheta):
					for j in range(resPhi):
						self.tau0[i,j,k,l] = int( np.argmin( np.abs( self.IMTDF[i,j,k,l]-tau )) )
		self.tau0[self.tau0>K//2] -= K
		self.tau0 = self.tau0.transpose([2, 3, 0, 1])
	
	def forward(self, x):
		tau0 = self.tau0.copy()
		tau0[tau0<0] += x.shape[-1]
		maps = torch.zeros(list(x.shape[0:-3]) + [self.resTheta, self.resPhi], device=x.device).float()
		for n in range(self.N):
			for m in range(self.N):
				maps += x[..., n, m, tau0[n,m,:,:]]
		
		if self.avoid_negatives: maps = torch.relu(maps)
		
		if self.normalize:
			maps -= torch.mean( torch.mean(maps, -1, keepdim=True), -2, keepdim=True)
			maps += 1e-12 # To avoid numerical issues
			maps /= torch.max(torch.max(maps, -1, keepdim=True)[0], -2, keepdim=True)[0]
			
		return maps


class SRP_grid_map(nn.Module):
	""" Compute the SRP-PHAT maps for a given grid from the GCCs taken as input.
	In the constructor of the layer, you need to indicate the number of signals (N) and the window length (K),
	the desired gird for the map (grid), the microphone positions relative to the center of the
	array (rn) and the sampling frequency (fs).
	With normalize=True (default) each map is normalized to ethe range [-1,1] approximately
	"""
	def __init__(self, N, K, grid, rn, fs, c=343.0, normalize=True, avoid_negatives=False):
		super(SRP_grid_map, self).__init__()

		self.N = N
		self.K = K
		self.grid = grid
		self.fs = float(fs)
		self.normalize = normalize
		self.avoid_negatives = avoid_negatives

		tau0 = np.empty(self.grid.shape[:-1] + (N, N), dtype=int)
		tau = np.concatenate([range(0, K//2+1), range(-K//2+1, 0)])/float(fs)  # Valid discrete values
		for k in range(N):
			for l in range(N):
				IMTDF = np.dot( self.grid, rn[l,:]-rn[k,:] ) / c  # Time differences, floats
				tau0[..., k, l] = np.argmin(np.abs(IMTDF[..., np.newaxis] - tau), -1)  # Time differences, samples
		tau0[tau0>K//2] -= K
		self.tau0 = einops.rearrange(tau0, "... Ni Nj -> Ni Nj ...", Ni=N, Nj=N)

	def forward(self, x):
		tau0 = self.tau0.copy()
		tau0[tau0<0] += x.shape[-1]
		maps = torch.zeros(x.shape[0:-3] + self.grid.shape[:-1], device=x.device).float()
		for n in range(self.N):
			for m in range(self.N):
				maps += x[..., n, m, tau0[n,m,...]]

		if self.avoid_negatives: maps = torch.relu(maps)

		if self.normalize:
			original_shape = maps.shape
			maps += 1e-12 # To avoid numerical issues
			maps = maps.reshape(maps.shape[:-self.grid.ndim+1] + (-1,))
			maps = maps / maps.amax(-1, keepdim=True)
			maps = maps.reshape(original_shape)

		return maps

class SRP_icosahedral_map(SRP_grid_map):
	""" Compute the SRP-PHAT icosahedral maps from the GCCs taken as input.
	In the constructor of the layer, you need to indicate the number of signals (N) and the window length (K), the
	desired resolution of the icosahedron (r), the microphone positions relative to the center of the
	array (rn) and the sampling frequency (fs).
	With normalize=True (default) each map is normalized to ethe range [-1,1] approximately
	"""
	def __init__(self, N, K, r, rn, fs, c=343.0, normalize=True, avoid_negatives=False):
		grid = icoCNN.icosahedral_grid_coordinates(r)
		super(SRP_icosahedral_map, self).__init__(N, K, grid, rn, fs, c, normalize, avoid_negatives)


class SphericPad(nn.Module):
	""" Replication padding for time axis, reflect padding for the elevation and circular padding for the azimuth.
	The time padding is optional, do not use it with CausConv3d.
	"""
	def __init__(self, pad):
		super(SphericPad, self).__init__()
		
		if len(pad) == 4:
			self.padLeft, self.padRight, self.padTop, self.padBottom = pad
			self.padFront, self.padBack = 0, 0
		elif len(pad) == 6:
			self.padLeft, self.padRight, self.padTop, self.padBottom, self.padFront, self.padBack = pad
		else: 
			raise Exception('Expect 4 or 6 values for padding (padLeft, padRight, padTop, padBottom, [padFront, padBack])')
	
	def forward(self, x):
		assert x.shape[-1] >= self.padRight and x.shape[-1] >= self.padLeft, \
			'Padding size should be less than the corresponding input dimension for the azimuth axis'

		if self.padBack > 0 or self.padFront > 0:
			x = F.pad(x, (0, 0, 0, 0, self.padFront, self.padBack), 'replicate')

		input_shape = x.shape
		x = x.view((x.shape[0], -1, x.shape[-2], x.shape[-1]))

		x = F.pad(x, (0, 0, self.padTop, self.padBottom), 'reflect') # Actually, it should add a pi shift

		x = torch.cat((x[..., -self.padLeft:], x, x[..., :self.padRight]), dim=-1)

		return x.view((x.shape[0],) + input_shape[1:-2] + (x.shape[-2], x.shape[-1]))


