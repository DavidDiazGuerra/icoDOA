"""
	Utils functions to deal with spherical coordinates in Pytorch.

	File name: utils.py
	Author: David Diaz-Guerra
	Date creation: 03/2022
	Python Version: 3.8.1
	Pytorch Version: 1.8.1
"""

from math import pi
import torch
import numpy as np


def cart2sph(cart, include_r=False):
	""" Cartesian coordinates to spherical coordinates conversion.
	Each row contains one point in format (x, y, x) or (elevation, azimuth, radius),
	where the radius is optional according to the include_r argument.
	"""
	r = torch.sqrt(torch.sum(torch.pow(cart, 2), dim=-1))
	theta = torch.acos(cart[..., 2] / (r +1e-12))
	phi = torch.atan2(cart[..., 1], cart[..., 0])
	if include_r:
		sph = torch.stack((theta, phi, r), dim=-1)
	else:
		sph = torch.stack((theta, phi), dim=-1)
	return sph


def sph2cart(sph):
	""" Spherical coordinates to cartesian coordinates conversion.
	Each row contains one point in format (x, y, x) or (elevation, azimuth, radius),
	where the radius is supposed to be 1 if it is not included.
	"""
	if sph.shape[-1] == 2: sph = torch.cat((sph, torch.ones_like(sph[..., 0]).unsqueeze(-1)), dim=-1)
	x = sph[..., 2] * torch.sin(sph[..., 0]) * torch.cos(sph[..., 1])
	y = sph[..., 2] * torch.sin(sph[..., 0]) * torch.sin(sph[..., 1])
	z = sph[..., 2] * torch.cos(sph[..., 0])
	return torch.stack((x, y, z), dim=-1)


def angular_error(the_pred, phi_pred, the_true, phi_true):
	""" Angular distance between spherical coordinates.
	"""
	aux = torch.cos(the_true) * torch.cos(the_pred) + \
		  torch.sin(the_true) * torch.sin(the_pred) * torch.cos(phi_true - phi_pred)

	return torch.acos(torch.clamp(aux, -0.99999, 0.99999))


def mean_square_angular_error(y_pred, y_true):
	""" Mean square angular distance between spherical coordinates.
	Each row contains one point in format (elevation, azimuth).
	"""
	the_true = y_true[..., 0]
	phi_true = y_true[..., 1]
	the_pred = y_pred[..., 0]
	phi_pred = y_pred[..., 1]

	return torch.mean(torch.pow(angular_error(the_pred, phi_pred, the_true, phi_true), 2), -1)


def rms_angular_error_deg(y_pred, y_true):
	""" Root mean square angular distance between spherical coordinates.
	Each input row contains one point in format (elevation, azimuth) in radians
	but the output is in degrees.
	"""
	return torch.sqrt(mean_square_angular_error(y_pred, y_true)) * 180 / pi

def local_maxima_finder(maps, coor_grid, number, size=20*pi/180, threshold=0.0):
	loc_max_maps = torch.zeros_like(maps)
	abs_max_val = maps.view((maps.shape[:-2] + (-1,))).max(-1)[0]
	for t in range(coor_grid.shape[1]):
		for p in range(coor_grid.shape[2]):
			mask = angular_error(coor_grid[0,...], coor_grid[1,...], coor_grid[0,t,p], coor_grid[1,t,p]) < size
			# angular_error fails for very small distnaces (never returns 0 but 0.0045) so do not use very small size
			# actually, it should never be smaller than the bigger step in coor_grid
			# mask[t,p] = 0 # Two adjacent max values would be lost
			max_values = (maps * mask.float()).view(maps.shape[:-2] + (-1,)).max(dim=-1)[0]
			loc_max_maps[:, :, t, p] = maps[:, :, t, p] * (max_values == maps[..., t, p]).float()

	top_max_val, top_max_idx = loc_max_maps.view(maps.shape[0:2] + (-1,)).topk(number)
	t_idx, p_idx = np.unravel_index(top_max_idx.detach().cpu(), maps.shape[-2:])
	t_idx, p_idx = torch.from_numpy(t_idx), torch.from_numpy(p_idx)
	t_coor, p_corr = coor_grid[0, t_idx, 0], coor_grid[1, 0, p_idx]
	t_coor[(top_max_val < threshold*abs_max_val.unsqueeze(-1).expand(top_max_val.shape))] = float('nan')
	p_corr[(top_max_val < threshold*abs_max_val.unsqueeze(-1).expand(top_max_val.shape))] = float('nan')
	return torch.stack((t_coor, p_corr), -1)

def prod(l):
	m = 1;
	for li in l:
		m *= li
	return m
