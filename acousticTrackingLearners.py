"""
	Learner classes to train the models and perform inferences.

	File name: acousticTrackingLearners.py
	Author: David Diaz-Guerra
	Date creation: 03/2022
	Python Version: 3.8.1
	Pytorch Version: 1.8.1
"""

import numpy as np
import torch
import torch.optim as optim
from abc import ABC, abstractmethod
from tqdm import trange
from itertools import permutations

from utils import sph2cart, cart2sph, rms_angular_error_deg, angular_error, local_maxima_finder
import acousticTrackingModules as at_modules

import acousticTrackingModels as at_models


class TrackingLearner(ABC):
	""" Abstract class for training acoustic source tracking models and performing inferences.
	"""

	def __init__(self, model, preprocessor):
		self.model = model
		self.preprocessor = preprocessor
		self.cuda_activated = False
		super().__init__()

	def cuda(self):
		""" Move the model to the GPU and perform the training and inference there.
		"""
		self.model.cuda()
		self.cuda_activated = True
		self.preprocessor.cuda_activated = True

	def cpu(self):
		""" Move the model back to the CPU and perform the training and inference here.
		"""
		self.model.cpu()
		self.cuda_activated = False
		self.preprocessor.cuda_activated = False

	def getNetworkInput_batch(self, mic_sig_batch, acoustic_scene_batch=None):
		""" Get the network input for a data batch
		"""
		if acoustic_scene_batch is not None:
			return self.preprocessor.data_transformation(mic_sig_batch, acoustic_scene_batch)[0].cpu().detach().numpy()
		else:
			return self.preprocessor.data_transformation(mic_sig_batch).cpu().detach().numpy()

	def getNetworkInput_dataset(self, dataset, trajectories_per_batch):
		""" Get the network input for a datataset
		"""
		for batch_idx in range(len(dataset) // trajectories_per_batch):
			mic_sig_batch, acoustic_scene_batch = dataset.get_batch(batch_idx * trajectories_per_batch,
																	(batch_idx + 1) * trajectories_per_batch)

			inputs_batch = self.getNetworkInput_batch(mic_sig_batch)
			if batch_idx == 0:
				inputs = np.empty((len(dataset), inputs_batch.shape[1], inputs_batch.shape[2], inputs_batch.shape[3]))
			inputs[batch_idx * trajectories_per_batch:(batch_idx + 1) * trajectories_per_batch, :, :, :] = inputs_batch

		return inputs

	@abstractmethod
	def train_epoch(self, dataset, trajectories_per_batch, trajectories_per_gpu_call=5, lr=0.001, epoch=None):
		""" To be implemented in each learner according to its training scheme.
		"""
		pass

	@abstractmethod
	def test_epoch(self, dataset, trajectories_per_batch, nb_batchs=None):
		""" To be implemented in each learner according to its test scheme.
		"""
		pass

	@abstractmethod
	def predict_batch(self, mic_sig_batch, return_x=False):
		""" To be implemented in each learner according to its test scheme.
		You can use return_x=True in order to return the input of the model in addition to the DOA estimation
		"""
		pass

	@abstractmethod
	def predict_dataset(self, dataset, trajectories_per_batch, nb_batchs=None, return_rmsae=False, save_x=False,
						x_filed_name='netInput'):
		""" To be implemented in each learner according to its test scheme.
		It outputs the analyzed AcousticScenes with the DOA estimation in the field DOAw_pred.
		It can also include a filed with input of the network (save_x) and return the RMSAE (return_rmsae).
		"""
		pass


class OneSourceTrackingLearner(TrackingLearner):
	""" Class with the routines to train the one source tracking models and perform inferences.
	"""
	def __init__(self, model, preprocessor):
		super().__init__(model, preprocessor)

	def train_epoch(self, dataset, trajectories_per_batch, trajectories_per_gpu_call=5, lr=0.001, epoch=None, sigma=1.0):
		""" Train the model with an epoch of the dataset.
		"""
		assert trajectories_per_batch % trajectories_per_gpu_call == 0

		avg_loss = 0
		avg_beta = 0.99

		self.model.train()  # set the model in "training mode"
		optimizer = optim.Adam(self.model.parameters(), lr=lr)

		n_trajectories = len(dataset)

		trajectory_idx = 0
		optimizer.zero_grad()
		pbar = trange(n_trajectories // trajectories_per_gpu_call, ascii=True)
		for gpu_call_idx in pbar:
			if epoch is not None: pbar.set_description('Epoch {}'.format(epoch + 1))

			mic_sig_batch, acoustic_scene_batch = dataset.get_batch(gpu_call_idx * trajectories_per_gpu_call,
																	(gpu_call_idx + 1) * trajectories_per_gpu_call)
			x_batch, DOA_batch = self.preprocessor.data_transformation(mic_sig_batch, acoustic_scene_batch)
			x_batch.requires_grad_()
			DOA_batch_pred_cart = self.model(x_batch).contiguous()
			
			# DOA_batch = DOA_batch[..., 5:, :]
			# DOA_batch_pred_cart = DOA_batch_pred_cart[..., 5:, :]
			
			DOA_batch = DOA_batch.contiguous()
			DOA_batch_cart = sigma * sph2cart(DOA_batch)
			loss = torch.nn.functional.mse_loss(DOA_batch_pred_cart.view(-1, 3), DOA_batch_cart.view(-1, 3))
			loss.backward()

			trajectory_idx += trajectories_per_gpu_call

			if trajectory_idx % trajectories_per_batch == 0:
				optimizer.step()
				optimizer.zero_grad()

			avg_loss = avg_beta * avg_loss + (1 - avg_beta) * loss.item()
			pbar.set_postfix(loss=avg_loss / (1 - avg_beta ** (gpu_call_idx + 1)))

		optimizer.step()
		optimizer.zero_grad()

		del DOA_batch_pred_cart, loss

	def test_epoch(self, dataset, trajectories_per_batch, nb_batchs=None, sigma=1.0):
		""" Test the model with an epoch of the dataset.
		"""
		self.model.eval()  # set the model in "testing mode"
		with torch.no_grad():
			loss_data = 0
			rmsae_data = 0

			n_trajectories = len(dataset)
			if nb_batchs is None:
				nb_batchs = n_trajectories // trajectories_per_batch

			for idx in range(nb_batchs):
				mic_sig_batch, acoustic_scene_batch = dataset.get_batch(idx * trajectories_per_batch,
																		(idx + 1) * trajectories_per_batch)
				x_batch, DOA_batch = self.preprocessor.data_transformation(mic_sig_batch, acoustic_scene_batch)
				DOA_batch_pred_cart = self.model(x_batch).contiguous()
				
				# DOA_batch = DOA_batch[..., 5:, :]
				# DOA_batch_pred_cart = DOA_batch_pred_cart[..., 5:, :]
				
				DOA_batch = DOA_batch.contiguous()
				DOA_batch_cart = sigma * sph2cart(DOA_batch)
				loss_data += torch.nn.functional.mse_loss(DOA_batch_pred_cart.view(-1, 3), DOA_batch_cart.view(-1, 3))

				DOA_batch_pred = cart2sph(DOA_batch_pred_cart)
				rmsae_data += rms_angular_error_deg(DOA_batch[..., 5:, :].view(-1, 2), DOA_batch_pred[..., 5:, :].view(-1, 2))

			loss_data /= nb_batchs
			rmsae_data /= nb_batchs

			return loss_data, rmsae_data

	def predict_batch(self, mic_sig_batch, vad_batch=None, return_x=False):
		""" Perform the model inference for an input batch.
		You can use return_x=True in order to return the input of the model in addition to the DOA estimation
		"""
		self.model.eval()  # set the model in "testing mode"

		n_trajectories = mic_sig_batch.shape[0]
		trajectory_len = mic_sig_batch.shape[1]

		x_batch = self.preprocessor.data_transformation(mic_sig_batch, vad_batch=vad_batch)

		DOA_batch_pred_cart = self.model(x_batch).cpu().detach()
		DOA_batch_pred = cart2sph(DOA_batch_pred_cart)

		if return_x:
			return DOA_batch_pred, x_batch.cpu().detach()
		else:
			return DOA_batch_pred

	def predict_dataset(self, dataset, trajectories_per_batch, nb_batchs=None, return_rmsae=False, save_x=False,
						x_filed_name='netInput'):
		""" Perform the model inference over the whole dataset.
		It outputs the analyzed AcousticScenes with the DOA estimation in the field DOAw_pred.
		It can also include a filed with input of the network (save_x) and return the RMSAE (return_rmsae).
		"""
		self.model.eval()  # set the model in "testing mode"
		with torch.no_grad():

			n_trajectories = len(dataset)
			if nb_batchs is None:
				nb_batchs = n_trajectories // trajectories_per_batch

			acoustic_scenes = []
			# rmsae = 0

			for idx in range(nb_batchs):
				mic_sig_batch, acoustic_scene_batch = dataset.get_batch(idx * trajectories_per_batch,
																		(idx + 1) * trajectories_per_batch)
				vad_batch = np.array([acoustic_scene_batch[i].vad for i in range(len(acoustic_scene_batch))])

				if save_x:
					DOA_batch_pred, x_batch = self.predict_batch(mic_sig_batch, vad_batch=vad_batch, return_x=True)
				else:
					DOA_batch_pred = self.predict_batch(mic_sig_batch, vad_batch=vad_batch, return_x=False)

				for i in range(len(acoustic_scene_batch)):
					# acoustic_scene_batch[i].DOAw_pred = [DOA_batch_pred[i,s,...].numpy() for s in range(DOA_batch_pred[i].shape[0])]
					acoustic_scene_batch[i].DOAw_pred = DOA_batch_pred[i,...].unsqueeze(0).numpy()
					if save_x:
						if x_filed_name == 'maps':  # Save only the map, not the other channels
							setattr(acoustic_scene_batch[i], x_filed_name, x_batch[i, 0, ...].numpy())
						else:
							setattr(acoustic_scene_batch[i], x_filed_name, x_batch[i, ...].numpy())
					acoustic_scenes.append(acoustic_scene_batch[i])

				# if return_rmsae:
				# 	DOA_batch = self.preprocessor.data_transformation(acoustic_scene_batch=acoustic_scene_batch).cpu().contiguous()
				#
				# 	batch_error = 0
				# 	for i in range(DOA_batch.shape[0]):
				# 		errors = []
				# 		for pairing in permutations(range(DOA_batch.shape[1])):
				# 			errors.append(rms_angular_error_deg(DOA_batch_pred[i, pairing, ...].view(-1, 2),
				# 													   DOA_batch[i,...].view(-1, 2)))
				# 		batch_error += min(errors)
				# 	rmsae += batch_error / DOA_batch.shape[0]

			if return_rmsae:
				rmsae = 0
				for acoustic_scene in acoustic_scenes:
					rmsae += acoustic_scene.get_rmsae(frames_to_exclude=5)
				return acoustic_scenes, rmsae / len(acoustic_scenes)
			else:
				return acoustic_scenes


class OneSourceClassificationLearner(TrackingLearner):
	""" Class with the routines to train the one source tracking models and perform inferences.
	"""

	def __init__(self, model, preprocessor, out_res_the, out_res_phi, arrayType='3D'):
		super().__init__(model, preprocessor)

		self.out_coor_grid = torch.zeros((2, out_res_the, out_res_phi))
		thetaMax = np.pi / 2 if arrayType == 'planar' else np.pi
		theta = torch.linspace(0, thetaMax, out_res_the).reshape((out_res_the,1))
		phi = torch.linspace(-np.pi, np.pi, out_res_phi+1)[:-1]
		self.out_coor_grid[0,...] = theta.repeat((1, out_res_phi))
		self.out_coor_grid[1,...] = phi.repeat((out_res_the, 1))
		if self.cuda_activated:
			self.out_coor_grid.cuda()

		self.sig = 10*np.pi/180

	def cuda(self):
		super().cuda()
		self.out_coor_grid = self.out_coor_grid.cuda()

	def cpu(self):
		super().cpu()
		self.out_coor_grid.cpu()

	def match_DOAs(self, output_max_coor):
		DOA_batch_pred = torch.zeros((output_max_coor.shape[:2] + (2,)), device=output_max_coor.device)
		for b in range(output_max_coor.shape[0]):
			DOA_batch_pred[b,0,:] = output_max_coor[b,0,0,:]
			for f in range(2, output_max_coor.shape[1]):
				track_distance = angular_error(output_max_coor[b,f,:,0], output_max_coor[b,f,:,1], DOA_batch_pred[b,f-1,0], DOA_batch_pred[b,f-1,1])
				track_distance[torch.isnan(track_distance)] = float('inf')
				DOA_batch_pred[b,f,:] = output_max_coor[b,f,track_distance.argmin(),:]

		return DOA_batch_pred

	def train_epoch(self, dataset, trajectories_per_batch, trajectories_per_gpu_call=5, lr=0.001, epoch=None):
		""" Train the model with an epoch of the dataset.
		"""
		assert trajectories_per_batch % trajectories_per_gpu_call == 0

		avg_loss = 0
		avg_beta = 0.99

		self.model.train()  # set the model in "training mode"
		optimizer = optim.Adam(self.model.parameters(), lr=lr)

		n_trajectories = len(dataset)

		trajectory_idx = 0
		optimizer.zero_grad()
		pbar = trange(n_trajectories // trajectories_per_gpu_call, ascii=True)
		for gpu_call_idx in pbar:
			if epoch is not None: pbar.set_description('Epoch {}'.format(epoch + 1))

			mic_sig_batch, acoustic_scene_batch = dataset.get_batch(gpu_call_idx * trajectories_per_gpu_call,
																	(gpu_call_idx + 1) * trajectories_per_gpu_call)
			x_batch, DOA_batch = self.preprocessor.data_transformation(mic_sig_batch, acoustic_scene_batch)
			x_batch = x_batch[:,0,...].unsqueeze(1)
			x_batch.requires_grad_()
			classification_output = self.model(x_batch)[1].squeeze()

			classification_reference = torch.zeros_like(classification_output)
			theta_grid = self.out_coor_grid[0, ...].expand(classification_reference.shape[1:])
			phi_grid = self.out_coor_grid[1, ...].expand(classification_reference.shape[1:])
			for b in range(DOA_batch.shape[0]):
				for s in range(DOA_batch.shape[1]):
					theta_ref = DOA_batch[b, s, :, 0].unsqueeze(-1).unsqueeze(-1).expand(theta_grid.shape)
					phi_ref = DOA_batch[b, s, :, 1].unsqueeze(-1).unsqueeze(-1).expand(phi_grid.shape)
					classification_reference[b,...] += torch.exp(-0.5*(angular_error(theta_grid, phi_grid, theta_ref, phi_ref))**2/(self.sig)**2)
					# angular_error fails for very small distnaces (never returns 0 but 0.0045) so do not use very small self.sig

			loss = torch.nn.functional.mse_loss(classification_output, classification_reference)
			loss.backward()

			trajectory_idx += trajectories_per_gpu_call

			if trajectory_idx % trajectories_per_batch == 0:
				optimizer.step()
				optimizer.zero_grad()

			avg_loss = avg_beta * avg_loss + (1 - avg_beta) * loss.item()
			pbar.set_postfix(loss=avg_loss / (1 - avg_beta ** (gpu_call_idx + 1)))

			pbar.update()

		optimizer.step()
		optimizer.zero_grad()

	def test_epoch(self, dataset, trajectories_per_batch, nb_batchs=None):
		pass
		""" Test the model with an epoch of the dataset.
		"""
		self.model.eval()  # set the model in "testing mode"
		with torch.no_grad():
			rmsae_data = 0

			n_trajectories = len(dataset)
			if nb_batchs is None:
				nb_batchs = n_trajectories // trajectories_per_batch

			for idx in range(nb_batchs):
				mic_sig_batch, acoustic_scene_batch = dataset.get_batch(idx * trajectories_per_batch,
																		(idx + 1) * trajectories_per_batch)
				x_batch, DOA_batch = self.preprocessor.data_transformation(mic_sig_batch, acoustic_scene_batch)
				x_batch = x_batch[:, 0, ...].unsqueeze(1)
				classification_output = self.model(x_batch)[1].squeeze()

				output_max_coor = local_maxima_finder(classification_output, self.out_coor_grid, 5, threshold=0.25)

				DOA_batch_pred = self.match_DOAs(output_max_coor)

				DOA_batch_best_prediction_ref = torch.zeros_like(DOA_batch_pred)
				for b in range(DOA_batch.shape[0]):
					rmasaes = torch.tensor([rms_angular_error_deg(DOA_batch[b,s,...], DOA_batch_pred[b,:]) for s in range(DOA_batch.shape[1])])
					DOA_batch_best_prediction_ref[b,...] = DOA_batch[b,rmasaes.argmin(),...]

				rmsae_data += rms_angular_error_deg(DOA_batch_best_prediction_ref.view(-1, 2), DOA_batch_pred.view(-1, 2))

			rmsae_data /= nb_batchs

			return rmsae_data

	def predict_batch(self, mic_sig_batch, acoustic_scene_batch, return_x=False):
		""" Perform the model inference for an input batch.
		You can use return_x=True in order to return the input of the model in addition to the DOA estimation
		"""
		self.model.eval()  # set the model in "testing mode"

		n_trajectories = mic_sig_batch.shape[0]
		trajectory_len = mic_sig_batch.shape[1]

		x_batch, DOA_batch = self.preprocessor.data_transformation(mic_sig_batch, acoustic_scene_batch)
		x_batch = x_batch[:, 0, ...].unsqueeze(1)
		classification_output = self.model(x_batch)[1].squeeze()

		output_max_coor = local_maxima_finder(classification_output, self.out_coor_grid, 5, threshold=0.25)
		DOA_batch_pred = self.match_DOAs(output_max_coor)

		if return_x:
			return DOA_batch_pred.cpu().detach(), x_batch.cpu().detach()
		else:
			return DOA_batch_pred.cpu().detach()

	def predict_dataset(self, dataset, trajectories_per_batch, nb_batchs=None, return_rmsae=False, save_x=False,
						x_filed_name='netInput'):
		""" Perform the model inference over the whole dataset.
		It outputs the analyzed AcousticScenes with the DOA estimation in the field DOAw_pred.
		It can also include a filed with input of the network (save_x) and return the RMSAE (return_rmsae).
		"""
		self.model.eval()  # set the model in "testing mode"

		n_trajectories = len(dataset)
		if nb_batchs is None:
			nb_batchs = n_trajectories // trajectories_per_batch

		acoustic_scenes = []
		# rmsae = 0

		for idx in range(nb_batchs):
			mic_sig_batch, acoustic_scene_batch = dataset.get_batch(idx * trajectories_per_batch,
																	(idx + 1) * trajectories_per_batch)

			if save_x:
				DOA_batch_pred, x_batch = self.predict_batch(mic_sig_batch, acoustic_scene_batch, return_x=True)
			else:
				DOA_batch_pred = self.predict_batch(mic_sig_batch, acoustic_scene_batch, return_x=False)

			for i in range(len(acoustic_scene_batch)):
				acoustic_scene_batch[i].DOAw_pred = [DOA_batch_pred[i, s, ...].numpy() for s in
													 range(DOA_batch_pred[i].shape[0])]
				if save_x:
					if x_filed_name == 'maps':  # Save only the map, not the other channels
						setattr(acoustic_scene_batch[i], x_filed_name, x_batch[i, 0, ...].numpy())
					else:
						setattr(acoustic_scene_batch[i], x_filed_name, x_batch[i, ...].numpy())
				acoustic_scenes.append(acoustic_scene_batch[i])

		if return_rmsae:
			rmsae = 0
			rmsae_best_source = 0
			for acoustic_scene in acoustic_scenes:
				rmsae += acoustic_scene.get_rmsae()
				rmsaes = [rms_angular_error_deg(torch.tensor(acoustic_scene.DOAw[s]),
												torch.tensor(acoustic_scene.DOAw_pred[s])) for s in
						  range(acoustic_scene.n_sources)]
				rmsae_best_source += min(rmsaes)
			return acoustic_scenes, rmsae / len(acoustic_scenes), rmsae_best_source / len(acoustic_scenes)
		else:
			return acoustic_scenes


class MultiSourceClassificationLearner(TrackingLearner):
	""" Class with the routines to train the one source tracking models and perform inferences.
	"""

	@staticmethod
	def best_source_error_per_frame(y, y_pred): # Angular error
		errors = torch.zeros(y.shape[:-1], device=y.device)
		for i in range(y.shape[-3]):
			errors[...,i,:] = angular_error(y[...,i,:,0], y[...,i,:,1], y_pred[...,0,:,0], y_pred[...,0,:,1])
		best_pairing_error = errors.min(-2)[0].mean()
		return best_pairing_error

	def __init__(self, model, preprocessor, out_res_the, out_res_phi, arrayType='3D'):
		super().__init__(model, preprocessor)

		self.out_coor_grid = torch.zeros((2, out_res_the, out_res_phi))
		thetaMax = np.pi / 2 if arrayType == 'planar' else np.pi
		theta = torch.linspace(0, thetaMax, out_res_the).reshape((out_res_the,1))
		phi = torch.linspace(-np.pi, np.pi, out_res_phi+1)[:-1]
		self.out_coor_grid[0,...] = theta.repeat((1, out_res_phi))
		self.out_coor_grid[1,...] = phi.repeat((out_res_the, 1))
		if self.cuda_activated:
			self.out_coor_grid.cuda()

		self.sig = 10*np.pi/180

	def cuda(self):
		super().cuda()
		self.out_coor_grid = self.out_coor_grid.cuda()

	def cpu(self):
		super().cpu()
		self.out_coor_grid.cpu()

	def match_DOAs(self, output_max_coor, DOA_batch):
		DOA_batch_trans = DOA_batch.transpose(1, 2)

		err1 = angular_error(output_max_coor[..., 0], output_max_coor[..., 1], DOA_batch_trans[..., 0],
							 DOA_batch_trans[..., 1])
		err2 = angular_error(output_max_coor.flip(-2)[..., 0], output_max_coor.flip(-2)[..., 1],
							 DOA_batch_trans[..., 0], DOA_batch_trans[..., 1])
		err1, err2 = err1.sum(-1), err2.sum(-1)

		DOA_batch_pred = torch.zeros_like(DOA_batch)
		for b in range(DOA_batch.shape[0]):
			for f in range(DOA_batch.shape[2]):
				DOA_batch_pred[b,:,f,:] = output_max_coor[b,f,:,:] if err1[b,f] < err2[b,f] else output_max_coor[b,f,:,:].flip(0)

		return DOA_batch_pred

	def train_epoch(self, dataset, trajectories_per_batch, trajectories_per_gpu_call=5, lr=0.001, epoch=None):
		""" Train the model with an epoch of the dataset.
		"""
		assert trajectories_per_batch % trajectories_per_gpu_call == 0

		avg_loss = 0
		avg_beta = 0.99

		self.model.train()  # set the model in "training mode"
		optimizer = optim.Adam(self.model.parameters(), lr=lr)

		n_trajectories = len(dataset)

		trajectory_idx = 0
		optimizer.zero_grad()
		pbar = trange(n_trajectories // trajectories_per_gpu_call, ascii=True)
		for gpu_call_idx in pbar:
			if epoch is not None: pbar.set_description('Epoch {}'.format(epoch + 1))

			mic_sig_batch, acoustic_scene_batch = dataset.get_batch(gpu_call_idx * trajectories_per_gpu_call,
																	(gpu_call_idx + 1) * trajectories_per_gpu_call)
			x_batch, DOA_batch = self.preprocessor.data_transformation(mic_sig_batch, acoustic_scene_batch)
			x_batch = x_batch[:,0,...].unsqueeze(1)
			x_batch.requires_grad_()
			classification_output = self.model(x_batch)[1].squeeze()

			classification_reference = torch.zeros_like(classification_output)
			theta_grid = self.out_coor_grid[0, ...].expand(classification_reference.shape[1:])
			phi_grid = self.out_coor_grid[1, ...].expand(classification_reference.shape[1:])
			for b in range(DOA_batch.shape[0]):
				for s in range(DOA_batch.shape[1]):
					theta_ref = DOA_batch[b, s, :, 0].unsqueeze(-1).unsqueeze(-1).expand(theta_grid.shape)
					phi_ref = DOA_batch[b, s, :, 1].unsqueeze(-1).unsqueeze(-1).expand(phi_grid.shape)
					classification_reference[b,...] += torch.exp(-0.5*(angular_error(theta_grid, phi_grid, theta_ref, phi_ref))**2/(self.sig)**2)
					# angular_error fails for very small distnaces (never returns 0 but 0.0045) so do not use very small self.sig

			loss = torch.nn.functional.mse_loss(classification_output, classification_reference)
			loss.backward()

			trajectory_idx += trajectories_per_gpu_call

			if trajectory_idx % trajectories_per_batch == 0:
				optimizer.step()
				optimizer.zero_grad()

			avg_loss = avg_beta * avg_loss + (1 - avg_beta) * loss.item()
			pbar.set_postfix(loss=avg_loss / (1 - avg_beta ** (gpu_call_idx + 1)))

			pbar.update()

		optimizer.step()
		optimizer.zero_grad()

	def test_epoch(self, dataset, trajectories_per_batch, nb_batchs=None):
		pass
		""" Test the model with an epoch of the dataset.
		"""
		self.model.eval()  # set the model in "testing mode"
		with torch.no_grad():
			rmsae_best_source = 0
			rmsae_all = 0

			n_trajectories = len(dataset)
			if nb_batchs is None:
				nb_batchs = n_trajectories // trajectories_per_batch

			for idx in range(nb_batchs):
				mic_sig_batch, acoustic_scene_batch = dataset.get_batch(idx * trajectories_per_batch,
																		(idx + 1) * trajectories_per_batch)
				x_batch, DOA_batch = self.preprocessor.data_transformation(mic_sig_batch, acoustic_scene_batch)
				x_batch = x_batch[:, 0, ...].unsqueeze(1)
				classification_output = self.model(x_batch)[1].squeeze()

				output_max_coor = local_maxima_finder(classification_output, self.out_coor_grid, DOA_batch.shape[1])

				DOA_batch_pred = self.match_DOAs(output_max_coor, DOA_batch)

				rmsae_best_source += torch.sqrt(torch.mean( rms_angular_error_deg(DOA_batch, DOA_batch_pred).min(1)[0] **2))
				rmsae_all += rms_angular_error_deg(DOA_batch.view(-1, 2), DOA_batch_pred.view(-1, 2))

			rmsae_best_source /= nb_batchs
			rmsae_all /= nb_batchs

			return rmsae_best_source, rmsae_all

	def predict_batch(self, mic_sig_batch, acoustic_scene_batch, return_x=False):
		""" Perform the model inference for an input batch.
		You can use return_x=True in order to return the input of the model in addition to the DOA estimation
		"""
		self.model.eval()  # set the model in "testing mode"

		n_trajectories = mic_sig_batch.shape[0]
		trajectory_len = mic_sig_batch.shape[1]

		x_batch, DOA_batch = self.preprocessor.data_transformation(mic_sig_batch, acoustic_scene_batch)
		x_batch = x_batch[:, 0, ...].unsqueeze(1)
		classification_output = self.model(x_batch)[1].squeeze()

		output_max_coor = local_maxima_finder(classification_output, self.out_coor_grid, DOA_batch.shape[1])
		DOA_batch_pred = self.match_DOAs(output_max_coor, DOA_batch)

		if return_x:
			return DOA_batch_pred.cpu().detach(), x_batch.cpu().detach()
		else:
			return DOA_batch_pred.cpu().detach()

	def predict_dataset(self, dataset, trajectories_per_batch, nb_batchs=None, return_rmsae=False, save_x=False,
						x_filed_name='netInput'):
		""" Perform the model inference over the whole dataset.
		It outputs the analyzed AcousticScenes with the DOA estimation in the field DOAw_pred.
		It can also include a filed with input of the network (save_x) and return the RMSAE (return_rmsae).
		"""
		self.model.eval()  # set the model in "testing mode"

		n_trajectories = len(dataset)
		if nb_batchs is None:
			nb_batchs = n_trajectories // trajectories_per_batch

		acoustic_scenes = []
		# rmsae = 0

		for idx in range(nb_batchs):
			mic_sig_batch, acoustic_scene_batch = dataset.get_batch(idx * trajectories_per_batch,
																	(idx + 1) * trajectories_per_batch)

			if save_x:
				DOA_batch_pred, x_batch = self.predict_batch(mic_sig_batch, acoustic_scene_batch, return_x=True)
			else:
				DOA_batch_pred = self.predict_batch(mic_sig_batch, acoustic_scene_batch, return_x=False)

			for i in range(len(acoustic_scene_batch)):
				acoustic_scene_batch[i].DOAw_pred = [DOA_batch_pred[i, s, ...].numpy() for s in
													 range(DOA_batch_pred[i].shape[0])]
				if save_x:
					if x_filed_name == 'maps':  # Save only the map, not the other channels
						setattr(acoustic_scene_batch[i], x_filed_name, x_batch[i, 0, ...].numpy())
					else:
						setattr(acoustic_scene_batch[i], x_filed_name, x_batch[i, ...].numpy())
				acoustic_scenes.append(acoustic_scene_batch[i])

		if return_rmsae:
			rmsae = 0
			rmsae_best_source = 0
			for acoustic_scene in acoustic_scenes:
				rmsae += acoustic_scene.get_rmsae()
				# rmsaes = [rms_angular_error_deg(torch.tensor(acoustic_scene.DOAw[s]),
				# 								torch.tensor(acoustic_scene.DOAw_pred[s])) for s in
				# 		  range(acoustic_scene.n_sources)]
				# rmsae_best_source += min(rmsaes)
				rmsae_best_source += self.best_source_error_per_frame(torch.tensor(acoustic_scene.DOAw),
																	  torch.tensor(acoustic_scene.DOAw_pred))
			return acoustic_scenes, rmsae / len(acoustic_scenes), rmsae_best_source / len(acoustic_scenes)
		else:
			return acoustic_scenes


class MultiSourceTrackingLearner(TrackingLearner):
	""" Abstract class with the routines to train the multisource tracking models and perform inferences.
	"""

	def __init__(self, model, preprocessor):
		super().__init__(model, preprocessor)

	@staticmethod
	def best_pairing_error(y, y_pred, error_func):
		errors = []
		for pairing in permutations(range(y.shape[0])):
			errors.append( error_func(y_pred[pairing, ...].view(-1, y_pred.shape[-1]), y.view(-1, y.shape[-1])) )
		return min(errors)

	@staticmethod
	def best_source_error_per_frame(y, y_pred): # Angular error
		errors = torch.zeros(y.shape[:-1], device=y.device)
		for i in range(y.shape[-3]):
			errors[...,i,:] = angular_error(y[...,i,:,0], y[...,i,:,1], y_pred[...,0,:,0], y_pred[...,0,:,1])
		best_pairing_error = errors.min(-2)[0].mean()
		return best_pairing_error

	def train_epoch(self, dataset, trajectories_per_batch, trajectories_per_gpu_call=5, lr=0.0001, epoch=None):
		""" Train the model with an epoch of the dataset.
		"""
		assert trajectories_per_batch % trajectories_per_gpu_call == 0

		avg_loss = 0
		avg_beta = 0.99

		self.model.train()  # set the model in "training mode"
		optimizer = optim.Adam(self.model.parameters(), lr=lr)

		n_trajectories = len(dataset)

		trajectory_idx = 0
		optimizer.zero_grad()
		pbar = trange(n_trajectories // trajectories_per_gpu_call, ascii=True)
		for gpu_call_idx in pbar:
			if epoch is not None: pbar.set_description('Epoch {}'.format(epoch + 1))

			mic_sig_batch, acoustic_scene_batch = dataset.get_batch(gpu_call_idx * trajectories_per_gpu_call,
																	(gpu_call_idx + 1) * trajectories_per_gpu_call)
			x_batch, DOA_batch = self.preprocessor.data_transformation(mic_sig_batch, acoustic_scene_batch)
			x_batch.requires_grad_()
			DOA_batch_pred_cart = self.model(x_batch).contiguous()

			DOA_batch = DOA_batch.contiguous()
			DOA_batch_cart = sph2cart(DOA_batch)

			loss = 0.0
			for i in range(DOA_batch_cart.shape[0]):
				loss += self.best_pairing_error(DOA_batch_cart[i, ...], DOA_batch_pred_cart[i, ...],
												torch.nn.functional.mse_loss)
			loss /= DOA_batch_cart.shape[0]
			loss = torch.nn.functional.mse_loss(DOA_batch_cart, DOA_batch_pred_cart)
			loss.backward()

			trajectory_idx += trajectories_per_gpu_call

			if trajectory_idx % trajectories_per_batch == 0:
				optimizer.step()
				optimizer.zero_grad()

			avg_loss = avg_beta * avg_loss + (1 - avg_beta) * loss.item()
			pbar.set_postfix(loss=avg_loss / (1 - avg_beta ** (gpu_call_idx + 1)))

			pbar.update()

		optimizer.step()
		optimizer.zero_grad()

		del DOA_batch_pred_cart, loss

	def test_epoch(self, dataset, trajectories_per_batch, nb_batchs=None):
		""" Test the model with an epoch of the dataset.
		"""
		self.model.eval()  # set the model in "testing mode"
		with torch.no_grad():
			loss_data = 0
			rmsae_data = 0

			n_trajectories = len(dataset)
			if nb_batchs is None:
				nb_batchs = n_trajectories // trajectories_per_batch

			for idx in range(nb_batchs):
				mic_sig_batch, acoustic_scene_batch = dataset.get_batch(idx * trajectories_per_batch,
																		(idx + 1) * trajectories_per_batch)
				x_batch, DOA_batch = self.preprocessor.data_transformation(mic_sig_batch, acoustic_scene_batch)
				DOA_batch_pred_cart = self.model(x_batch).contiguous()

				DOA_batch = DOA_batch.contiguous()
				DOA_batch_cart = sph2cart(DOA_batch)

				batch_loss = 0
				for i in range(DOA_batch_cart.shape[0]):
					batch_loss += self.best_pairing_error(DOA_batch_cart[i, ...], DOA_batch_pred_cart[i, ...],
														  torch.nn.functional.mse_loss)
				loss_data += batch_loss / DOA_batch_cart.shape[0]

				DOA_batch_pred = cart2sph(DOA_batch_pred_cart)

				batch_error = 0
				for i in range(DOA_batch.shape[0]):
					batch_error += self.best_pairing_error(DOA_batch[i, ...], DOA_batch_pred[i, ...],
														   rms_angular_error_deg)
				rmsae_data += batch_error / DOA_batch.shape[0]

			loss_data /= nb_batchs
			rmsae_data /= nb_batchs

			return loss_data, rmsae_data

	def predict_batch(self, mic_sig_batch, return_x=False):
		""" Perform the model inference for an input batch.
		You can use return_x=True in order to return the input of the model in addition to the DOA estimation
		"""
		self.model.eval()  # set the model in "testing mode"

		n_trajectories = mic_sig_batch.shape[0]
		trajectory_len = mic_sig_batch.shape[1]

		x_batch = self.preprocessor.data_transformation(mic_sig_batch)

		DOA_batch_pred_cart = self.model(x_batch).cpu().detach()
		DOA_batch_pred = cart2sph(DOA_batch_pred_cart)

		if return_x:
			return DOA_batch_pred, x_batch.cpu().detach()
		else:
			return DOA_batch_pred

	def predict_dataset(self, dataset, trajectories_per_batch, nb_batchs=None, return_rmsae=False, save_x=False,
						x_filed_name='netInput'):
		""" Perform the model inference over the whole dataset.
		It outputs the analyzed AcousticScenes with the DOA estimation in the field DOAw_pred.
		It can also include a filed with input of the network (save_x) and return the RMSAE (return_rmsae).
		"""
		self.model.eval()  # set the model in "testing mode"

		n_trajectories = len(dataset)
		if nb_batchs is None:
			nb_batchs = n_trajectories // trajectories_per_batch

		acoustic_scenes = []
		# rmsae = 0

		for idx in range(nb_batchs):
			mic_sig_batch, acoustic_scene_batch = dataset.get_batch(idx * trajectories_per_batch,
																	(idx + 1) * trajectories_per_batch)

			if save_x:
				DOA_batch_pred, x_batch = self.predict_batch(mic_sig_batch, return_x=True)
			else:
				DOA_batch_pred = self.predict_batch(mic_sig_batch, return_x=False)

			for i in range(len(acoustic_scene_batch)):
				acoustic_scene_batch[i].DOAw_pred = [DOA_batch_pred[i, s, ...].numpy() for s in
													 range(DOA_batch_pred[i].shape[0])]
				if save_x:
					if x_filed_name == 'maps':  # Save only the map, not the other channels
						setattr(acoustic_scene_batch[i], x_filed_name, x_batch[i, 0, ...].numpy())
					else:
						setattr(acoustic_scene_batch[i], x_filed_name, x_batch[i, ...].numpy())
				acoustic_scenes.append(acoustic_scene_batch[i])

		# if return_rmsae:
		# 	DOA_batch = self.preprocessor.data_transformation(acoustic_scene_batch=acoustic_scene_batch).cpu().contiguous()
		#
		# 	batch_error = 0
		# 	for i in range(DOA_batch.shape[0]):
		# 		errors = []
		# 		for pairing in permutations(range(DOA_batch.shape[1])):
		# 			errors.append(rms_angular_error_deg(DOA_batch_pred[i, pairing, ...].view(-1, 2),
		# 													   DOA_batch[i,...].view(-1, 2)))
		# 		batch_error += min(errors)
		# 	rmsae += batch_error / DOA_batch.shape[0]

		if return_rmsae:
			rmsae = 0
			rmsae_best_source = 0
			for acoustic_scene in acoustic_scenes:
				rmsae += acoustic_scene.get_rmsae()
				rmsae_best_source += self.best_source_error_per_frame(torch.tensor(acoustic_scene.DOAw),
																	  torch.tensor(acoustic_scene.DOAw_pred))
			return acoustic_scenes, rmsae / len(acoustic_scenes), rmsae_best_source / len(acoustic_scenes)
		else:
			return acoustic_scenes


class RecursiveMultiSourceTrackingLearner(TrackingLearner):
	""" Abstract class with the routines to train the multisource tracking models and perform inferences.
	"""

	def __init__(self, model, preprocessor, arrayType='3D'):
		super().__init__(model, preprocessor)

		self.res_the = preprocessor.res_the
		self.res_phi = preprocessor.res_phi

		cr_deep = int(min(4, np.log2(min(self.res_the, self.res_phi))))
		self.classification_model = at_models.UCross3D(self.res_the, self.res_phi, input_channels=1, cr_deep=cr_deep)
		self.classification_model.load_state_dict(torch.load('models/2sourceTracking_UCross3D_robot_K4096_16x32_classification_model.bin'))

		self.coor_grid = torch.zeros((2, self.res_the, self.res_phi))
		thetaMax = np.pi / 2 if arrayType == 'planar' else np.pi
		theta = torch.linspace(0, thetaMax, self.res_the).reshape((self.res_the,1))
		phi = torch.linspace(-np.pi, np.pi, self.res_phi+1)[:-1]
		self.coor_grid[0,...] = theta.repeat((1, self.res_phi))
		self.coor_grid[1,...] = phi.repeat((self.res_the, 1))

		if self.cuda_activated:
			self.classification_model.cuda()
			self.coor_grid.cuda()


	def cuda(self):
		super().cuda()
		self.classification_model.cuda()
		self.coor_grid = self.coor_grid.cuda()

	def cpu(self):
		super().cpu()
		self.classification_model.cpu()
		self.coor_grid = self.coor_grid.cpu()

	@staticmethod
	def best_pairing_error(y, y_pred, error_func):
		errors = []
		for pairing in permutations(range(y.shape[0])):
			errors.append( error_func(y_pred[pairing, ...].view(-1, y_pred.shape[-1]), y.view(-1, y.shape[-1])) )
		return min(errors)

	@staticmethod
	def best_pairing_error_per_frame(y, y_pred): #MSE
		errors = torch.zeros(y.shape[:-1], device=y.device)
		for i in range(y.shape[1]):
			errors[:,i,:] = torch.sqrt(torch.sum((y[:, i, ...] - y_pred[:, 0, ...])**2, -1))
		best_errors = errors.min(1)[0]
		return (best_errors**2).mean()

	@staticmethod
	def best_DOA_pairing_per_frame(y, y_pred): # Angular error
		errors = torch.zeros(y.shape[:-1], device=y.device)
		for i in range(y.shape[1]):
			errors[:,i,:] = angular_error(y[:,i,:,0], y[:,i,:,1], y_pred[:,0,:,0], y_pred[:,0,:,1])
		best_pairing = errors.min(1)[1]
		return best_pairing

	def train_epoch(self, dataset, trajectories_per_batch, trajectories_per_gpu_call=5, lr=0.0001, epoch=None):
		""" Train the model with an epoch of the dataset.
		"""
		assert trajectories_per_batch % trajectories_per_gpu_call == 0

		avg_loss = 0
		avg_beta = 0.99

		self.model.train()  # set the model in "training mode"
		optimizer = optim.Adam(self.model.parameters(), lr=lr)

		n_trajectories = len(dataset)

		trajectory_idx = 0
		optimizer.zero_grad()
		pbar = trange(n_trajectories // trajectories_per_gpu_call, ascii=True)
		for gpu_call_idx in pbar:
			if epoch is not None: pbar.set_description('Epoch {}'.format(epoch + 1))

			mic_sig_batch, acoustic_scene_batch = dataset.get_batch(gpu_call_idx * trajectories_per_gpu_call,
																	(gpu_call_idx + 1) * trajectories_per_gpu_call)
			x_batch, DOA_batch = self.preprocessor.data_transformation(mic_sig_batch.sum(1), acoustic_scene_batch)

			classification_output = self.classification_model(x_batch)[1]  #TODO: Tiene gradientes???
			cancellation_guide = (classification_output == classification_output.max(-1)[0].max(-1)[0].unsqueeze(-1).unsqueeze(-1)).float()
			x_batch = torch.cat((x_batch, cancellation_guide), 1)

			x_batch.requires_grad_()
			DOA_output, cancellation_output = DOA_batch_pred_cart = self.model(x_batch)

			class_max_coor = local_maxima_finder(cancellation_guide.squeeze(), self.coor_grid, 1).transpose(1,2)
			classified_sourece = self.best_DOA_pairing_per_frame(DOA_batch, class_max_coor)
			classified_sourece = classified_sourece.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
			classified_sourece_expanded = classified_sourece.expand(classified_sourece.shape[:-2]+(self.res_the, self.res_phi))

			n_sources = acoustic_scene_batch[0].n_sources.item()
			cancelled_maps = torch.zeros((x_batch.shape[0], n_sources) + x_batch.shape[2:], device=x_batch.device)
			for n in range(n_sources):
				active_sources = list(range(n_sources))
				active_sources.remove(n)
				cancelled_maps[:,n,...] = self.preprocessor.data_transformation(mic_sig_batch[:,active_sources,...].sum(1)).squeeze()
			cancellation_goal = torch.gather(cancelled_maps, 1, classified_sourece_expanded)

			loss = torch.nn.functional.mse_loss(cancellation_output, cancellation_goal)
			loss.backward()

			# DOA_batch = DOA_batch.contiguous()
			# DOA_batch_cart = sph2cart(DOA_batch)
			#
			# loss = self.best_pairing_error_per_frame(DOA_batch_cart, DOA_batch_pred_cart)
			# loss.backward()

			trajectory_idx += trajectories_per_gpu_call

			if trajectory_idx % trajectories_per_batch == 0:
				optimizer.step()
				optimizer.zero_grad()

			avg_loss = avg_beta * avg_loss + (1 - avg_beta) * loss.item()
			pbar.set_postfix(loss=avg_loss / (1 - avg_beta ** (gpu_call_idx + 1)))

			pbar.update()

		optimizer.step()
		optimizer.zero_grad()

		del DOA_batch_pred_cart, loss

	def test_epoch(self, dataset, trajectories_per_batch, nb_batchs=None):
		""" Test the model with an epoch of the dataset.
		"""
		self.model.eval()  # set the model in "testing mode"
		with torch.no_grad():
			loss_data = 0
			rmsae_data = 0

			n_trajectories = len(dataset)
			if nb_batchs is None:
				nb_batchs = n_trajectories // trajectories_per_batch

			for idx in range(nb_batchs):
				mic_sig_batch, acoustic_scene_batch = dataset.get_batch(idx * trajectories_per_batch,
																		(idx + 1) * trajectories_per_batch)
				x_batch, DOA_batch = self.preprocessor.data_transformation(mic_sig_batch, acoustic_scene_batch)
				DOA_batch_pred_cart = self.model(x_batch).contiguous()

				DOA_batch = DOA_batch.contiguous()
				DOA_batch_cart = sph2cart(DOA_batch)

				batch_loss = 0
				for i in range(DOA_batch_cart.shape[0]):
					batch_loss += self.best_pairing_error(DOA_batch_cart[i, ...], DOA_batch_pred_cart[i, ...],
														  torch.nn.functional.mse_loss)
				loss_data += batch_loss / DOA_batch_cart.shape[0]

				DOA_batch_pred = cart2sph(DOA_batch_pred_cart)

				batch_error = 0
				for i in range(DOA_batch.shape[0]):
					batch_error += self.best_pairing_error(DOA_batch[i, ...], DOA_batch_pred[i, ...],
														   rms_angular_error_deg)
				rmsae_data += batch_error / DOA_batch.shape[0]

			loss_data /= nb_batchs
			rmsae_data /= nb_batchs

			return loss_data, rmsae_data

	def predict_batch(self, mic_sig_batch, return_x=False):
		""" Perform the model inference for an input batch.
		You can use return_x=True in order to return the input of the model in addition to the DOA estimation
		"""
		self.model.eval()  # set the model in "testing mode"

		n_trajectories = mic_sig_batch.shape[0]
		trajectory_len = mic_sig_batch.shape[1]

		x_batch = self.preprocessor.data_transformation(mic_sig_batch)

		DOA_batch_pred_cart = self.model(x_batch).cpu().detach()
		DOA_batch_pred = cart2sph(DOA_batch_pred_cart)

		if return_x:
			return DOA_batch_pred, x_batch.cpu().detach()
		else:
			return DOA_batch_pred

	def predict_dataset(self, dataset, trajectories_per_batch, nb_batchs=None, return_rmsae=False, save_x=False,
						x_filed_name='netInput'):
		""" Perform the model inference over the whole dataset.
		It outputs the analyzed AcousticScenes with the DOA estimation in the field DOAw_pred.
		It can also include a filed with input of the network (save_x) and return the RMSAE (return_rmsae).
		"""
		self.model.eval()  # set the model in "testing mode"

		n_trajectories = len(dataset)
		if nb_batchs is None:
			nb_batchs = n_trajectories // trajectories_per_batch

		acoustic_scenes = []
		# rmsae = 0

		for idx in range(nb_batchs):
			mic_sig_batch, acoustic_scene_batch = dataset.get_batch(idx * trajectories_per_batch,
																	(idx + 1) * trajectories_per_batch)

			if save_x:
				DOA_batch_pred, x_batch = self.predict_batch(mic_sig_batch, return_x=True)
			else:
				DOA_batch_pred = self.predict_batch(mic_sig_batch, return_x=False)

			for i in range(len(acoustic_scene_batch)):
				acoustic_scene_batch[i].DOAw_pred = [DOA_batch_pred[i, s, ...].numpy() for s in
													 range(DOA_batch_pred[i].shape[0])]
				if save_x:
					if x_filed_name == 'maps':  # Save only the map, not the other channels
						setattr(acoustic_scene_batch[i], x_filed_name, x_batch[i, 0, ...].numpy())
					else:
						setattr(acoustic_scene_batch[i], x_filed_name, x_batch[i, ...].numpy())
				acoustic_scenes.append(acoustic_scene_batch[i])

		# if return_rmsae:
		# 	DOA_batch = self.preprocessor.data_transformation(acoustic_scene_batch=acoustic_scene_batch).cpu().contiguous()
		#
		# 	batch_error = 0
		# 	for i in range(DOA_batch.shape[0]):
		# 		errors = []
		# 		for pairing in permutations(range(DOA_batch.shape[1])):
		# 			errors.append(rms_angular_error_deg(DOA_batch_pred[i, pairing, ...].view(-1, 2),
		# 													   DOA_batch[i,...].view(-1, 2)))
		# 		batch_error += min(errors)
		# 	rmsae += batch_error / DOA_batch.shape[0]

		if return_rmsae:
			rmsae = 0
			for acoustic_scene in acoustic_scenes:
				rmsae += acoustic_scene.get_rmsae()
			return acoustic_scenes, rmsae / len(acoustic_scenes)
		else:
			return acoustic_scenes


# class RecursiveMultiSourceTrackingLearner(TrackingLearner):
# 	""" Abstract class with the routines to train the recursive multisource tracking models and perform inferences.
# 	"""
#
# 	def __init__(self, model, preprocessor, supervise_recursive_output=True, n_sources_known=True):
# 		assert supervise_recursive_output
# 		assert n_sources_known
# 		super().__init__(model, preprocessor)
#
# 		self.supervise_recursive_output = supervise_recursive_output
# 		self.n_sources_known = n_sources_known
#
# 	@staticmethod
# 	def best_pairing_error(y, y_pred, error_func):
# 		errors = []
# 		for pairing in permutations(range(y.shape[0])):
# 			errors.append( error_func(y_pred[pairing, ...].view(-1, y_pred.shape[-1]), y.view(-1, y.shape[-1])) )
# 		return min(errors)
#
# 	def train_epoch(self, dataset, trajectories_per_batch, trajectories_per_gpu_call=5, lr=0.001, epoch=None):
# 		""" Train the model with an epoch of the dataset.
# 		"""
# 		assert trajectories_per_batch % trajectories_per_gpu_call == 0
#
# 		avg_loss_DOA = 0
# 		avg_loss_maps = 0
# 		avg_loss_maps2 = 0
# 		avg_beta = 0.99
#
# 		self.model.train()  # set the model in "training mode"
# 		optimizer = optim.Adam(self.model.parameters(), lr=lr)
#
# 		n_trajectories = len(dataset)
#
# 		trajectory_idx = 0
# 		optimizer.zero_grad()
# 		pbar = trange(n_trajectories // trajectories_per_gpu_call, ascii=True)
# 		for gpu_call_idx in pbar:
# 			if epoch is not None: pbar.set_description('Epoch {}'.format(epoch + 1))
#
# 			mic_sig_batch, acoustic_scene_batch = dataset.get_batch(gpu_call_idx * trajectories_per_gpu_call,
# 																	(gpu_call_idx + 1) * trajectories_per_gpu_call)
#
# 			DOA_batch = self.preprocessor.data_transformation(None, acoustic_scene_batch)
# 			DOA_batch = DOA_batch.contiguous()
# 			DOA_batch_cart = sph2cart(DOA_batch)
# 			DOA_batch_pred_cart = torch.zeros_like(DOA_batch_cart)
#
# 			n_sources = acoustic_scene_batch[0].n_sources
# 			recursive_outputs, recursive_outputs2, recursive_outputs_ref, recursive_outputs_ref2 = [], [], [], []
# 			for i in range(n_sources):
# 				if i == 0:
# 					x_batch = self.preprocessor.data_transformation(mic_sig_batch[:, i:, ...].sum(axis=1))
# 				else:
# 					x_batch = self.preprocessor.apply_extras(recursive_output_batch)
# 				x_batch.requires_grad_()
# 				DOA_si_batch_pred_cart, recursive_output_batch = self.model(x_batch)
# 				# DOA_si_batch_pred_cart, recursive_output_batch, recursive_output_batch2 = self.model(x_batch)
# 				DOA_batch_pred_cart[:, i, ...] = DOA_si_batch_pred_cart.squeeze()
#
# 				for batch_idx in range(trajectories_per_gpu_call):
# 					min_error = float('inf')
# 					best_pairing = -1
# 					for j in range(i, n_sources):
# 						error = torch.nn.functional.mse_loss(DOA_si_batch_pred_cart[batch_idx, ...].squeeze(), DOA_batch_cart[batch_idx, j, ...])
# 						if error < min_error:
# 							best_pairing = j
# 							min_error = error
# 					aux = np.copy(mic_sig_batch[batch_idx, i, ...])
# 					mic_sig_batch[batch_idx, i, ...] = mic_sig_batch[batch_idx, best_pairing, ...]
# 					mic_sig_batch[batch_idx, best_pairing, ...] = aux
# 					aux = DOA_batch_cart[batch_idx, i, ...].clone()
# 					DOA_batch_cart[batch_idx, i, ...] = DOA_batch_cart[batch_idx, best_pairing, ...]
# 					DOA_batch_cart[batch_idx, best_pairing, ...] = aux
#
# 				if self.supervise_recursive_output and i < n_sources-1:
# 					recursive_outputs.append(recursive_output_batch.squeeze())
# 					# recursive_outputs2.append(recursive_output_batch2.squeeze())
# 					# recursive_outputs_ref2.append(self.preprocessor.data_transformation(mic_sig_batch[:, i, ...])[:, 0, ...])
# 				if self.supervise_recursive_output and i > 0:
# 					# recursive_outputs_ref.append(x_batch[:, 0, ...] if recursive_output_batch.shape[1]==1 else x_batch)
# 					recursive_outputs_ref.append(self.preprocessor.data_transformation(mic_sig_batch[:, i:, ...].sum(axis=1))[:,0,...])
#
# 			recursive_outputs = torch.stack(recursive_outputs)
# 			# recursive_outputs2 = torch.stack(recursive_outputs2)
# 			recursive_outputs_ref = torch.stack(recursive_outputs_ref)
# 			# recursive_outputs_ref2 = torch.stack(recursive_outputs_ref2)
# 			loss_DOA = torch.nn.functional.mse_loss(DOA_batch_pred_cart, DOA_batch_cart)
# 			loss_maps = 100 * torch.nn.functional.mse_loss(recursive_outputs, recursive_outputs_ref)
# 			# loss_maps2 = 100 * torch.nn.functional.mse_loss(recursive_outputs2, recursive_outputs_ref2)
# 			loss = loss_DOA #+ loss_maps #+ loss_maps2
# 			loss.backward()
#
# 			trajectory_idx += trajectories_per_gpu_call
#
# 			if trajectory_idx % trajectories_per_batch == 0:
# 				optimizer.step()
# 				optimizer.zero_grad()
#
# 			avg_loss_DOA = avg_beta * avg_loss_DOA + (1 - avg_beta) * loss_DOA.item()
# 			avg_loss_maps = avg_beta * avg_loss_maps + (1 - avg_beta) * loss_maps.item()
# 			# avg_loss_maps2 = avg_beta * avg_loss_maps2 + (1 - avg_beta) * loss_maps2.item()
# 			pbar.set_postfix(loss_DOA=avg_loss_DOA / (1 - avg_beta ** (gpu_call_idx + 1)),
# 							 loss_maps=avg_loss_maps / (1 - avg_beta ** (gpu_call_idx + 1)),
# 							 # loss_maps2=avg_loss_maps2 / (1 - avg_beta ** (gpu_call_idx + 1))
# 							 )
#
# 			pbar.update()
#
# 		optimizer.step()
# 		optimizer.zero_grad()
#
# 		del DOA_batch_pred_cart, loss
#
# 	def test_epoch(self, dataset, trajectories_per_batch, nb_batchs=None):
# 		""" Test the model with an epoch of the dataset.
# 		"""
# 		self.model.eval()  # set the model in "testing mode"
# 		with torch.no_grad():
# 			loss_data = 0
# 			rmsae_data = 0
#
# 			n_trajectories = len(dataset)
# 			if nb_batchs is None:
# 				nb_batchs = n_trajectories // trajectories_per_batch
#
# 			for idx in range(nb_batchs):
# 				mic_sig_batch, acoustic_scene_batch = dataset.get_batch(idx * trajectories_per_batch,
# 																		(idx + 1) * trajectories_per_batch)
# 				x_batch, DOA_batch = self.preprocessor.data_transformation(mic_sig_batch, acoustic_scene_batch)
# 				DOA_batch = DOA_batch.contiguous()
# 				DOA_batch_cart = sph2cart(DOA_batch)
#
# 				n_sources = acoustic_scene_batch[0].n_sources
# 				DOA_batch_pred_cart = []
# 				for i in range(n_sources):
# 					DOA_si_batch_pred_cart, recursive_output_batch = self.model(x_batch)
# 					# DOA_si_batch_pred_cart, recursive_output_batch, trash = self.model(x_batch)
# 					DOA_batch_pred_cart.append(DOA_si_batch_pred_cart)
# 					x_batch = self.preprocessor.apply_extras(recursive_output_batch)
# 				DOA_batch_pred_cart = torch.cat(DOA_batch_pred_cart, axis=1)
#
# 				batch_loss = 0
# 				for i in range(DOA_batch_cart.shape[0]):
# 					batch_loss += self.best_pairing_error(DOA_batch_cart[i, ...], DOA_batch_pred_cart[i, ...],
# 														  torch.nn.functional.mse_loss)
# 				loss_data += batch_loss / DOA_batch_cart.shape[0]
#
# 				DOA_batch_pred = cart2sph(DOA_batch_pred_cart)
# 				batch_error = 0
# 				for i in range(DOA_batch.shape[0]):
# 					batch_error += self.best_pairing_error(DOA_batch[i, ...], DOA_batch_pred[i, ...],
# 														   rms_angular_error_deg)
# 				rmsae_data += batch_error / DOA_batch.shape[0]
#
# 			loss_data /= nb_batchs
# 			rmsae_data /= nb_batchs
#
# 			return loss_data, rmsae_data
#
# 	def predict_batch(self, mic_sig_batch, return_x=False):
# 		""" Perform the model inference for an input batch.
# 		You can use return_x=True in order to return the input of the model in addition to the DOA estimation
# 		"""
# 		self.model.eval()  # set the model in "testing mode"
#
# 		n_trajectories = mic_sig_batch.shape[0]
# 		trajectory_len = mic_sig_batch.shape[1]
#
# 		x_batch = self.preprocessor.data_transformation(mic_sig_batch)
#
# 		DOA_batch_pred_cart = self.model(x_batch).cpu().detach()
# 		DOA_batch_pred = cart2sph(DOA_batch_pred_cart)
#
# 		if return_x:
# 			return DOA_batch_pred, x_batch.cpu().detach()
# 		else:
# 			return DOA_batch_pred
#
# 	def predict_dataset(self, dataset, trajectories_per_batch, nb_batchs=None, return_rmsae=False, save_x=False,
# 						x_filed_name='netInput'):
# 		""" Perform the model inference over the whole dataset.
# 		It outputs the analyzed AcousticScenes with the DOA estimation in the field DOAw_pred.
# 		It can also include a filed with input of the network (save_x) and return the RMSAE (return_rmsae).
# 		"""
# 		self.model.eval()  # set the model in "testing mode"
#
# 		n_trajectories = len(dataset)
# 		if nb_batchs is None:
# 			nb_batchs = n_trajectories // trajectories_per_batch
#
# 		acoustic_scenes = []
# 		# rmsae = 0
#
# 		for idx in range(nb_batchs):
# 			mic_sig_batch, acoustic_scene_batch = dataset.get_batch(idx * trajectories_per_batch,
# 																	(idx + 1) * trajectories_per_batch)
#
# 			if save_x:
# 				DOA_batch_pred, x_batch = self.predict_batch(mic_sig_batch, return_x=True)
# 			else:
# 				DOA_batch_pred = self.predict_batch(mic_sig_batch, return_x=False)
#
# 			for i in range(len(acoustic_scene_batch)):
# 				acoustic_scene_batch[i].DOAw_pred = [DOA_batch_pred[i, s, ...].numpy() for s in
# 													 range(DOA_batch_pred[i].shape[0])]
# 				if save_x:
# 					if x_filed_name == 'maps':  # Save only the map, not the other channels
# 						setattr(acoustic_scene_batch[i], x_filed_name, x_batch[i, 0, ...].numpy())
# 					else:
# 						setattr(acoustic_scene_batch[i], x_filed_name, x_batch[i, ...].numpy())
# 				acoustic_scenes.append(acoustic_scene_batch[i])
#
# 		# if return_rmsae:
# 		# 	DOA_batch = self.preprocessor.data_transformation(acoustic_scene_batch=acoustic_scene_batch).cpu().contiguous()
# 		#
# 		# 	batch_error = 0
# 		# 	for i in range(DOA_batch.shape[0]):
# 		# 		errors = []
# 		# 		for pairing in permutations(range(DOA_batch.shape[1])):
# 		# 			errors.append(rms_angular_error_deg(DOA_batch_pred[i, pairing, ...].view(-1, 2),
# 		# 													   DOA_batch[i,...].view(-1, 2)))
# 		# 		batch_error += min(errors)
# 		# 	rmsae += batch_error / DOA_batch.shape[0]
#
# 		if return_rmsae:
# 			rmsae = 0
# 			for acoustic_scene in acoustic_scenes:
# 				rmsae += acoustic_scene.get_rmsae()
# 			return acoustic_scenes, rmsae / len(acoustic_scenes)
# 		else:
# 			return acoustic_scenes
#

class Preprocessor(ABC):
	def __init__(self):
		self.cuda_activated = False

	@abstractmethod
	def data_transformation(self, mic_sig_batch=None, acoustic_scene_batch=None):
		''' Transform the mic signals to the desired network input and extract the DOA from the AcousticScene
		'''
		pass

	@abstractmethod
	def apply_extras(self, x):
		''' Apply any optional transformation the network input
		'''
		pass


class TrackingFromMapsPreprocessor(Preprocessor):
	""" Preprocessor for models which use SRP-PHAT maps as input
	"""
	def __init__(self, N, K, res_the, res_phi, rn, fs, c=343.0, arrayType='planar', cat_maxCoor=False, apply_vad=False):
		"""
		N: Number of microphones in the array
		K: Window size for the SRP-PHAT map computation
		res_the: Resolution of the maps in the elevation axis
		res_phi: Resolution of the maps in the azimuth axis
		rn: Position of each microphone relative to te center of the array
		fs: Sampling frequency
		c: Speed of the sound [default: 343.0]
		arrayType: 'planar' or '3D' whether all the microphones are in the same plane (and the maximum DOA elevation is pi/2) or not [default: 'planar']
		cat_maxCoor: Include to the network input tow addition channels with the normalized coordinates of each map maximum [default: False]
		apply_vad: Turn to zero all the map pixels in frames without speech signal [default: False]
		"""
		super().__init__()

		self.N = N
		self.K = K
		self.fs = fs
		self.res_the = res_the
		self.res_phi = res_phi

		self.cat_maxCoor = cat_maxCoor
		self.apply_vad = apply_vad

		dist_max = np.max([np.max([np.linalg.norm(rn[n, :] - rn[m, :]) for m in range(N)]) for n in range(N)])
		tau_max = int(np.ceil(dist_max / c * fs))
		self.gcc = at_modules.GCC(N, K, tau_max=tau_max, transform='PHAT')
		self.srp = at_modules.SRP_map(N, K, res_the, res_phi, rn, fs,
									  thetaMax=np.pi / 2 if arrayType == 'planar' else np.pi)

	def data_transformation(self, mic_sig_batch=None, acoustic_scene_batch=None, vad_batch=None):
		""" Compute the SRP-PHAT maps from the microphone signals and extract the DoA groundtruth from the AcousticScene
		"""
		output = []

		if mic_sig_batch is not None:
			mic_sig_batch = torch.from_numpy( mic_sig_batch.astype(np.float32) )
			mic_sig_batch = mic_sig_batch.unsqueeze(1) # Add channel axis

			if self.cuda_activated:
				mic_sig_batch = mic_sig_batch.cuda()

			maps = self.srp(self.gcc(mic_sig_batch))
			maps = self.apply_extras(maps, acoustic_scene_batch, vad_batch)

			output += [ maps ]

		if acoustic_scene_batch is not None:
			DOAw_batch = torch.stack([torch.tensor([acoustic_scene_batch[i].DOAw[n].astype(np.float32) for n in range(len(acoustic_scene_batch[i].DOAw))]) for i in range(len(acoustic_scene_batch))])
			if self.cuda_activated:
				DOAw_batch = DOAw_batch.cuda()
			output += [ DOAw_batch ]

		return output[0] if len(output)==1 else output

	def apply_extras(self, maps, acoustic_scene_batch, vad_batch=None):
		if self.cat_maxCoor:
			maximums = maps.view(list(maps.shape[:-2]) + [-1]).argmax(dim=-1)
			max_the = (maximums / self.res_phi).float() / maps.shape[-2]
			max_phi = (maximums % self.res_phi).float() / maps.shape[-1]
			repeat_factor = np.array(maps.shape)
			repeat_factor[:-2] = 1
			maps = torch.cat((maps,
							  max_the[..., None, None].repeat(repeat_factor.tolist()),
							  max_phi[..., None, None].repeat(repeat_factor.tolist())
							  ), 1)
		if self.apply_vad:
			if acoustic_scene_batch is not None:
				vad_batch = np.array([acoustic_scene_batch[i].vad for i in range(len(acoustic_scene_batch))])
			assert vad_batch is not None # Breaks if neither acoustic_scene_batch nor vad_batch was given
			vad_output_th = vad_batch.mean(axis=-1) > 2 / 3
			vad_output_th = vad_output_th[:, np.newaxis, :, np.newaxis, np.newaxis]
			vad_output_th = torch.from_numpy(vad_output_th.astype(float)).to(maps.device)
			repeat_factor = np.array(maps.shape)
			repeat_factor[:-2] = 1
			maps *= vad_output_th.float().repeat(repeat_factor.tolist())
		return maps


class TrackingFromIcoMapsPreprocessor(Preprocessor):
	""" Preprocessor for models which use SRP-PHAT maps as input
	"""
	def __init__(self, N, K, r, rn, fs, c=343.0, apply_vad=False):
		"""
		N: Number of microphones in the array
		K: Window size for the SRP-PHAT map computation
		r: Resolution of the icosahedral maps
		rn: Position of each microphone relative to te center of the array
		fs: Sampling frequency
		c: Speed of the sound [default: 343.0]
		apply_vad: Turn to zero all the map pixels in frames without speech signal [default: False]
		"""
		super().__init__()

		self.N = N
		self.K = K
		self.fs = fs
		self.r = r

		self.apply_vad = apply_vad

		dist_max = np.max([np.max([np.linalg.norm(rn[n, :] - rn[m, :]) for m in range(N)]) for n in range(N)])
		tau_max = int(np.ceil(dist_max / c * fs))
		self.gcc = at_modules.GCC(N, K, tau_max=tau_max, transform='PHAT')
		self.srp = at_modules.SRP_icosahedral_map(N, K, r, rn, fs)

	def data_transformation(self, mic_sig_batch=None, acoustic_scene_batch=None, vad_batch=None):
		""" Compute the SRP-PHAT maps from the microphone signals and extract the DoA groundtruth from the AcousticScene
		"""
		output = []

		if mic_sig_batch is not None:
			mic_sig_batch = torch.from_numpy( mic_sig_batch.astype(np.float32) )
			mic_sig_batch = mic_sig_batch.unsqueeze(1) # Add channel axis

			if self.cuda_activated:
				mic_sig_batch = mic_sig_batch.cuda()

			maps = self.srp(self.gcc(mic_sig_batch))
			maps = self.apply_extras(maps, acoustic_scene_batch, vad_batch)

			output += [ maps ]

		if acoustic_scene_batch is not None:
			DOAw_batch = torch.tensor(np.stack([np.stack([acoustic_scene_batch[i].DOAw[n].astype(np.float32)
														  for n in range(len(acoustic_scene_batch[i].DOAw))])
												for i in range(len(acoustic_scene_batch))]))			
			if self.cuda_activated:
				DOAw_batch = DOAw_batch.cuda()
			output += [ DOAw_batch ]

		return output[0] if len(output)==1 else output

	def apply_extras(self, maps, acoustic_scene_batch, vad_batch=None):
		if self.apply_vad:
			if acoustic_scene_batch is not None:
				vad_batch =  np.array([acoustic_scene_batch[i].vad for i in range(len(acoustic_scene_batch))])
			assert vad_batch is not None # Breaks if neither acoustic_scene_batch nor vad_batch was given
			vad_output_th = vad_batch.mean(axis=-1) > 2 / 3
			vad_output_th = vad_output_th[:, np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
			vad_output_th = torch.from_numpy(vad_output_th.astype(float)).to(maps.device)
			repeat_factor = np.array(maps.shape)
			repeat_factor[:-3] = 1
			maps *= vad_output_th.float().repeat(repeat_factor.tolist())
		return maps


class TrackingFromMaximumsPreprocessor(Preprocessor):
	""" Preprocessor for models which use the coordinates of the maximums of the SRP-PHAT maps as input
	"""
	def __init__(self, N, K, res_the, res_phi, rn, fs, c=343.0, arrayType='planar'):
		"""
		N: Number of microphones in the array
		K: Window size for the SRP-PHAT map computation
		res_the: Resolution of the maps in the elevation axis
		res_phi: Resolution of the maps in the azimuth axis
		rn: Position of each microphone relative to te center of the array
		fs: Sampling frequency
		c: Speed of the sound [default: 343.0]
		arrayType: 'planar' or '3D' whether all the microphones are in the same plane (and the maximum DOA elevation is pi/2) or not [default: 'planar']
		"""
		super().__init__()

		self.N = N
		self.K = K
		self.res_the = res_the
		self.res_phi = res_phi

		dist_max = np.max([np.max([np.linalg.norm(rn[n, :] - rn[m, :]) for m in range(N)]) for n in range(N)])
		tau_max = int(np.ceil(dist_max / c * fs))
		self.gcc = at_modules.GCC(N, K, tau_max=tau_max, transform='PHAT')
		self.srp = at_modules.SRP_map(N, K, res_the, res_phi, rn, fs,
									  thetaMax=np.pi / 2 if arrayType == 'planar' else np.pi)

	def data_transformation(self, mic_sig_batch=None, acoustic_scene_batch=None):
		""" Get the coordinates of the maximums of the SRP-PHAT maps and extract the DoA groundtruth from the AcousticScene
		"""
		output = []

		if mic_sig_batch is not None:
			mic_sig_batch = torch.from_numpy( mic_sig_batch.astype(np.float32) )
			mic_sig_batch = mic_sig_batch.view((-1, 1) + tuple(mic_sig_batch.shape)[1:]) # Add channel dimension
			if self.cuda_activated:
				mic_sig_batch = mic_sig_batch.cuda()
			maps = self.srp(self.gcc(mic_sig_batch))
			maximums = maps.view([maps.shape[0], maps.shape[2], -1]).argmax(dim=-1)
			max_the = (maximums / self.res_phi).float() / maps.shape[-2]
			max_phi = (maximums % self.res_phi).float() / maps.shape[-1]
			x_batch = torch.stack((max_the, max_phi), dim=-1)
			x_batch.transpose_(1, 2)

			output += [ x_batch ]

		if acoustic_scene_batch is not None:
			DOAw_batch = torch.stack([torch.tensor(
				[acoustic_scene_batch[i].DOAw[n].astype(np.float32) for n in range(len(acoustic_scene_batch[i].DOAw))])
									  for i in range(len(acoustic_scene_batch))])
			if self.cuda_activated:
				DOAw_batch = DOAw_batch.cuda()
			output += [DOAw_batch]

		return output[0] if len(output)==1 else output

	def apply_extras(self, x):
		return x


class TrackingFromGCCsPreprocessor(Preprocessor):
	""" Preprocessor for models which use the sequence of the Generalized Cross-Correlation functions as input
	"""
	def __init__(self, N, K, rn, fs, c=343.0):
		"""
		N: Number of microphones in the array
		K: Window size for the SRP-PHAT map computation
		rn: Position of each microphone relative to te center of the array (to get the needed length of the GCC)
		fs: Sampling frequency
		c: Speed of the sound [default: 343.0]
		"""
		super().__init__()

		self.N = N
		self.K = K

		self.nb_pairs = (self.N*(self.N-1))//2
		self.pair_idx = []
		for i in range(N):
			for j in range(i+1,N):
				self.pair_idx.append((i,j))

		dist_max = np.max([np.max([np.linalg.norm(rn[n, :] - rn[m, :]) for m in range(N)]) for n in range(N)])
		self.tau_max = int(np.ceil(dist_max / c * fs))
		self.gcc = at_modules.GCC(N, K, tau_max=self.tau_max, transform='PHAT')

	def data_transformation(self, mic_sig_batch=None, acoustic_scene_batch=None):
		""" Get the GCC sequence and extract the DoA groundtruth from the AcousticScene
		"""
		output = []

		if mic_sig_batch is not None:
			mic_sig_batch = torch.from_numpy( mic_sig_batch.astype(np.float32) )
			mic_sig_batch = mic_sig_batch.view((-1, 1) + tuple(mic_sig_batch.shape)[1:]) # Add channel dimension
			if self.cuda_activated:
				mic_sig_batch = mic_sig_batch.cuda()
			gccs_batch = self.gcc(mic_sig_batch)

			x_batch = torch.empty((gccs_batch.shape[0], gccs_batch.shape[2], self.nb_pairs, self.tau_max*2+1)).cuda()
			for i in range(self.nb_pairs):
				x_batch[:,:,i,:] = gccs_batch[:, 0, :, self.pair_idx[i][0], self.pair_idx[i][1], :]
			x_batch = x_batch.reshape((gccs_batch.shape[0], gccs_batch.shape[2], self.nb_pairs*(self.tau_max*2+1)))
			x_batch.transpose_(1,2)
			x_batch += 1e-9  # To avoid numerical issues
			x_batch /= torch.max(x_batch, -2, keepdim=True)[0]

			output += [x_batch]

		if acoustic_scene_batch is not None:
			DOAw_batch = torch.stack([torch.tensor(
				[acoustic_scene_batch[i].DOAw[n].astype(np.float32) for n in range(len(acoustic_scene_batch[i].DOAw))])
									  for i in range(len(acoustic_scene_batch))])
			if self.cuda_activated:
				DOAw_batch = DOAw_batch.cuda()
			output += [DOAw_batch]

		return output[0] if len(output)==1 else output

	def apply_extras(self, x):
		return x


class TrackingFromSpectrogramPreprocessor(Preprocessor):
	""" Preprocessor for models which use the spectrogram of each microphone signalas input
	"""
	def __init__(self, N, K):
		"""
		N: Number of microphones in the array
		K: Window size for the SRP-PHAT map computation
		"""
		super().__init__()

		self.N = N
		self.K = K

		self.nb_pairs = (self.N*(self.N-1))//2
		self.pair_idx = []
		for i in range(N):
			for j in range(i+1,N):
				self.pair_idx.append((i,j))

	def data_transformation(self, mic_sig_batch=None, acoustic_scene_batch=None):
		""" Compute the spectrogram of each microphone signal and extract the DoA groundtruth from the AcousticScene
		"""
		output = []

		if mic_sig_batch is not None:
			mic_sig_batch = torch.from_numpy( mic_sig_batch.astype(np.float32) )
			mic_sig_batch = mic_sig_batch.view((-1, 1) + tuple(mic_sig_batch.shape)[1:]) # Add channel dimension
			if self.cuda_activated:
				mic_sig_batch = mic_sig_batch.cuda()

			mic_sig_fft = torch.rfft(mic_sig_batch, 1) # torch.Size([5, 1, 103, 12, 2049, 2])
			spect = at_modules.complex_cart2polar(mic_sig_fft)
			spect[..., 0] /= spect[..., 0].max(dim=4, keepdim=True)[0]
			spect[..., 1] /= np.pi
			x_batch = spect.permute(0,1,3,5,2,4).reshape((mic_sig_batch.shape[0], -1, mic_sig_batch.shape[2], mic_sig_batch.shape[-1]//2+1)) # torch.Size([5, 24, 103, 2049])
			x_batch = x_batch[..., 1:] # Remove f=0, torch.Size([5, 24, 103, 2048])

			output += [x_batch]

		if acoustic_scene_batch is not None:
			DOAw_batch = torch.stack([torch.tensor(
				[acoustic_scene_batch[i].DOAw[n].astype(np.float32) for n in range(len(acoustic_scene_batch[i].DOAw))])
									  for i in range(len(acoustic_scene_batch))])
			if self.cuda_activated:
				DOAw_batch = DOAw_batch.cuda()
			output += [DOAw_batch]

		return output[0] if len(output)==1 else output

	def apply_extras(self, x):
		return x


class MapSeparationLearner(TrackingLearner):
	""" Abstract class with the routines to train the multisource tracking models and perform inferences.
	"""

	def __init__(self, model, preprocessor):
		super().__init__(model, preprocessor)

	@staticmethod
	def best_pairing_error(y, y_pred, error_func):
		errors = []
		for pairing in permutations(range(y.shape[0])):
			errors.append( error_func(y_pred[pairing, ...].view(-1, y_pred.shape[-1]), y.view(-1, y.shape[-1])) )
		return min(errors)

	def train_epoch(self, dataset, trajectories_per_batch, trajectories_per_gpu_call=5, lr=0.001, epoch=None):
		""" Train the model with an epoch of the dataset.
		"""
		assert trajectories_per_batch % trajectories_per_gpu_call == 0

		avg_loss = 0
		avg_beta = 0.99

		self.model.train()  # set the model in "training mode"
		optimizer = optim.Adam(self.model.parameters(), lr=lr)

		n_trajectories = len(dataset)

		trajectory_idx = 0
		optimizer.zero_grad()
		pbar = trange(n_trajectories // trajectories_per_gpu_call, ascii=True)
		for gpu_call_idx in pbar:
			if epoch is not None: pbar.set_description('Epoch {}'.format(epoch + 1))

			mic_sig_batch, acoustic_scene_batch = dataset.get_batch(gpu_call_idx * trajectories_per_gpu_call,
																	(gpu_call_idx + 1) * trajectories_per_gpu_call)
			map_original = self.preprocessor.data_transformation(mic_sig_batch.sum(axis=1))
			DOA_batch_pred_cart, map1, map2 = self.model(map_original)


			maps = torch.cat((map1, map2), 1)
			maps_ref = torch.zeros_like(maps)
			maps_ref[:, 0, ...] = self.preprocessor.data_transformation(mic_sig_batch[:, 0, ...])[:, 0, ...]
			maps_ref[:, 1, ...] = self.preprocessor.data_transformation(mic_sig_batch[:, 1, ...])[:, 0, ...]

			loss = 0.0
			for i in range(map1.shape[0]):
				loss += self.best_pairing_error(maps[i, ...], maps_ref[i, ...],
												torch.nn.functional.mse_loss)
			loss /= map1.shape[0]
			loss.backward()

			trajectory_idx += trajectories_per_gpu_call

			if trajectory_idx % trajectories_per_batch == 0:
				optimizer.step()
				optimizer.zero_grad()

			avg_loss = avg_beta * avg_loss + (1 - avg_beta) * loss.item()
			pbar.set_postfix(loss=avg_loss / (1 - avg_beta ** (gpu_call_idx + 1)))

			pbar.update()

		optimizer.step()
		optimizer.zero_grad()

		del DOA_batch_pred_cart, loss

	def test_epoch(self, dataset, trajectories_per_batch, nb_batchs=None):
		""" Test the model with an epoch of the dataset.
		"""
		self.model.eval()  # set the model in "testing mode"
		with torch.no_grad():
			loss_data = 0
			rmsae_data = 0

			n_trajectories = len(dataset)
			if nb_batchs is None:
				nb_batchs = n_trajectories // trajectories_per_batch

			for idx in range(nb_batchs):
				mic_sig_batch, acoustic_scene_batch = dataset.get_batch(idx * trajectories_per_batch,
																		(idx + 1) * trajectories_per_batch)
				x_batch, DOA_batch = self.preprocessor.data_transformation(mic_sig_batch, acoustic_scene_batch)
				DOA_batch_pred_cart = self.model(x_batch).contiguous()

				DOA_batch = DOA_batch.contiguous()
				DOA_batch_cart = sph2cart(DOA_batch)

				batch_loss = 0
				for i in range(DOA_batch_cart.shape[0]):
					batch_loss += self.best_pairing_error(DOA_batch_cart[i, ...], DOA_batch_pred_cart[i, ...],
														  torch.nn.functional.mse_loss)
				loss_data += batch_loss / DOA_batch_cart.shape[0]

				DOA_batch_pred = cart2sph(DOA_batch_pred_cart)

				batch_error = 0
				for i in range(DOA_batch.shape[0]):
					batch_error += self.best_pairing_error(DOA_batch[i, ...], DOA_batch_pred[i, ...],
														   rms_angular_error_deg)
				rmsae_data += batch_error / DOA_batch.shape[0]

			loss_data /= nb_batchs
			rmsae_data /= nb_batchs

			return loss_data, rmsae_data

	def predict_batch(self, mic_sig_batch, return_x=False):
		""" Perform the model inference for an input batch.
		You can use return_x=True in order to return the input of the model in addition to the DOA estimation
		"""
		self.model.eval()  # set the model in "testing mode"

		n_trajectories = mic_sig_batch.shape[0]
		trajectory_len = mic_sig_batch.shape[1]

		x_batch = self.preprocessor.data_transformation(mic_sig_batch)

		DOA_batch_pred_cart = self.model(x_batch).cpu().detach()
		DOA_batch_pred = cart2sph(DOA_batch_pred_cart)

		if return_x:
			return DOA_batch_pred, x_batch.cpu().detach()
		else:
			return DOA_batch_pred

	def predict_dataset(self, dataset, trajectories_per_batch, nb_batchs=None, return_rmsae=False, save_x=False,
						x_filed_name='netInput'):
		""" Perform the model inference over the whole dataset.
		It outputs the analyzed AcousticScenes with the DOA estimation in the field DOAw_pred.
		It can also include a filed with input of the network (save_x) and return the RMSAE (return_rmsae).
		"""
		self.model.eval()  # set the model in "testing mode"

		n_trajectories = len(dataset)
		if nb_batchs is None:
			nb_batchs = n_trajectories // trajectories_per_batch

		acoustic_scenes = []
		# rmsae = 0

		for idx in range(nb_batchs):
			mic_sig_batch, acoustic_scene_batch = dataset.get_batch(idx * trajectories_per_batch,
																	(idx + 1) * trajectories_per_batch)

			if save_x:
				DOA_batch_pred, x_batch = self.predict_batch(mic_sig_batch, return_x=True)
			else:
				DOA_batch_pred = self.predict_batch(mic_sig_batch, return_x=False)

			for i in range(len(acoustic_scene_batch)):
				acoustic_scene_batch[i].DOAw_pred = [DOA_batch_pred[i, s, ...].numpy() for s in
													 range(DOA_batch_pred[i].shape[0])]
				if save_x:
					if x_filed_name == 'maps':  # Save only the map, not the other channels
						setattr(acoustic_scene_batch[i], x_filed_name, x_batch[i, 0, ...].numpy())
					else:
						setattr(acoustic_scene_batch[i], x_filed_name, x_batch[i, ...].numpy())
				acoustic_scenes.append(acoustic_scene_batch[i])

		# if return_rmsae:
		# 	DOA_batch = self.preprocessor.data_transformation(acoustic_scene_batch=acoustic_scene_batch).cpu().contiguous()
		#
		# 	batch_error = 0
		# 	for i in range(DOA_batch.shape[0]):
		# 		errors = []
		# 		for pairing in permutations(range(DOA_batch.shape[1])):
		# 			errors.append(rms_angular_error_deg(DOA_batch_pred[i, pairing, ...].view(-1, 2),
		# 													   DOA_batch[i,...].view(-1, 2)))
		# 		batch_error += min(errors)
		# 	rmsae += batch_error / DOA_batch.shape[0]

		if return_rmsae:
			rmsae = 0
			for acoustic_scene in acoustic_scenes:
				rmsae += acoustic_scene.get_rmsae()
			return acoustic_scenes, rmsae / len(acoustic_scenes)
		else:
			return acoustic_scenes


class MapCancelatorLearner(TrackingLearner):
	""" Abstract class with the routines to train the multisource tracking models and perform inferences.
	"""

	def __init__(self, model, preprocessor):
		super().__init__(model, preprocessor)

	@staticmethod
	def best_pairing_error(y, y_pred, error_func):
		errors = []
		for pairing in permutations(range(y.shape[0])):
			errors.append( error_func(y_pred[pairing, ...].view(-1, y_pred.shape[-1]), y.view(-1, y.shape[-1])) )
		return min(errors)

	def train_epoch(self, dataset, trajectories_per_batch, trajectories_per_gpu_call=5, lr=0.001, epoch=None):
		""" Train the model with an epoch of the dataset.
		"""
		assert trajectories_per_batch % trajectories_per_gpu_call == 0

		avg_loss = 0
		avg_beta = 0.99

		self.model.train()  # set the model in "training mode"
		optimizer = optim.Adam(self.model.parameters(), lr=lr)

		n_trajectories = len(dataset)

		trajectory_idx = 0
		optimizer.zero_grad()
		pbar = trange(n_trajectories // trajectories_per_gpu_call, ascii=True)
		for gpu_call_idx in pbar:
			if epoch is not None: pbar.set_description('Epoch {}'.format(epoch + 1))

			mic_sig_batch, acoustic_scene_batch = dataset.get_batch(gpu_call_idx * trajectories_per_gpu_call,
																	(gpu_call_idx + 1) * trajectories_per_gpu_call)
			# mic_sig_batch[:, 1:-1, :20, ...] = 0
			map_original = self.preprocessor.data_transformation(mic_sig_batch.sum(axis=1))

			# map_original = map_original[:, 0, ...].unsqueeze(1)
			# map_original = map_original[:, 0:2, ...]
			# map_original[:, 1, ...] = 0
			map_original = torch.cat((map_original, torch.zeros_like(map_original)), dim=1)
			for i in range(map_original.shape[0]):
				el = acoustic_scene_batch[i].DOAw[0][:,0] / np.pi * (map_original.shape[-2]-1)
				az = (acoustic_scene_batch[i].DOAw[0][:,1] + np.pi) / (2*np.pi) * map_original.shape[-1]
				el, az = el.round(), az.round()
				az[az == map_original.shape[-1]] = 0
				for j in range(len(el)):
					map_original[i, 1, j, int(el[j]), int(az[j])] = 1

			DOA_batch_pred_cart, map_est = self.model(map_original)

			map_ref = self.preprocessor.data_transformation(mic_sig_batch[:, 1, ...])[:, 0, ...]

			loss = torch.nn.functional.mse_loss(map_est.squeeze(), map_ref)
			loss.backward()

			trajectory_idx += trajectories_per_gpu_call

			if trajectory_idx % trajectories_per_batch == 0:
				optimizer.step()
				optimizer.zero_grad()

			avg_loss = avg_beta * avg_loss + (1 - avg_beta) * loss.item()
			pbar.set_postfix(loss=avg_loss / (1 - avg_beta ** (gpu_call_idx + 1)))

			pbar.update()

		optimizer.step()
		optimizer.zero_grad()

		del DOA_batch_pred_cart, loss

	def test_epoch(self, dataset, trajectories_per_batch, nb_batchs=None):
		""" Test the model with an epoch of the dataset.
		"""
		self.model.eval()  # set the model in "testing mode"
		with torch.no_grad():
			loss_data = 0
			rmsae_data = 0

			n_trajectories = len(dataset)
			if nb_batchs is None:
				nb_batchs = n_trajectories // trajectories_per_batch

			for idx in range(nb_batchs):
				mic_sig_batch, acoustic_scene_batch = dataset.get_batch(idx * trajectories_per_batch,
																		(idx + 1) * trajectories_per_batch)
				x_batch, DOA_batch = self.preprocessor.data_transformation(mic_sig_batch, acoustic_scene_batch)
				DOA_batch_pred_cart = self.model(x_batch).contiguous()

				DOA_batch = DOA_batch.contiguous()
				DOA_batch_cart = sph2cart(DOA_batch)

				batch_loss = 0
				for i in range(DOA_batch_cart.shape[0]):
					batch_loss += self.best_pairing_error(DOA_batch_cart[i, ...], DOA_batch_pred_cart[i, ...],
														  torch.nn.functional.mse_loss)
				loss_data += batch_loss / DOA_batch_cart.shape[0]

				DOA_batch_pred = cart2sph(DOA_batch_pred_cart)

				batch_error = 0
				for i in range(DOA_batch.shape[0]):
					batch_error += self.best_pairing_error(DOA_batch[i, ...], DOA_batch_pred[i, ...],
														   rms_angular_error_deg)
				rmsae_data += batch_error / DOA_batch.shape[0]

			loss_data /= nb_batchs
			rmsae_data /= nb_batchs

			return loss_data, rmsae_data

	def predict_batch(self, mic_sig_batch, return_x=False):
		""" Perform the model inference for an input batch.
		You can use return_x=True in order to return the input of the model in addition to the DOA estimation
		"""
		self.model.eval()  # set the model in "testing mode"

		n_trajectories = mic_sig_batch.shape[0]
		trajectory_len = mic_sig_batch.shape[1]

		x_batch = self.preprocessor.data_transformation(mic_sig_batch)

		DOA_batch_pred_cart = self.model(x_batch).cpu().detach()
		DOA_batch_pred = cart2sph(DOA_batch_pred_cart)

		if return_x:
			return DOA_batch_pred, x_batch.cpu().detach()
		else:
			return DOA_batch_pred

	def predict_dataset(self, dataset, trajectories_per_batch, nb_batchs=None, return_rmsae=False, save_x=False,
						x_filed_name='netInput'):
		""" Perform the model inference over the whole dataset.
		It outputs the analyzed AcousticScenes with the DOA estimation in the field DOAw_pred.
		It can also include a filed with input of the network (save_x) and return the RMSAE (return_rmsae).
		"""
		self.model.eval()  # set the model in "testing mode"

		n_trajectories = len(dataset)
		if nb_batchs is None:
			nb_batchs = n_trajectories // trajectories_per_batch

		acoustic_scenes = []
		# rmsae = 0

		for idx in range(nb_batchs):
			mic_sig_batch, acoustic_scene_batch = dataset.get_batch(idx * trajectories_per_batch,
																	(idx + 1) * trajectories_per_batch)

			if save_x:
				DOA_batch_pred, x_batch = self.predict_batch(mic_sig_batch, return_x=True)
			else:
				DOA_batch_pred = self.predict_batch(mic_sig_batch, return_x=False)

			for i in range(len(acoustic_scene_batch)):
				acoustic_scene_batch[i].DOAw_pred = [DOA_batch_pred[i, s, ...].numpy() for s in
													 range(DOA_batch_pred[i].shape[0])]
				if save_x:
					if x_filed_name == 'maps':  # Save only the map, not the other channels
						setattr(acoustic_scene_batch[i], x_filed_name, x_batch[i, 0, ...].numpy())
					else:
						setattr(acoustic_scene_batch[i], x_filed_name, x_batch[i, ...].numpy())
				acoustic_scenes.append(acoustic_scene_batch[i])

		# if return_rmsae:
		# 	DOA_batch = self.preprocessor.data_transformation(acoustic_scene_batch=acoustic_scene_batch).cpu().contiguous()
		#
		# 	batch_error = 0
		# 	for i in range(DOA_batch.shape[0]):
		# 		errors = []
		# 		for pairing in permutations(range(DOA_batch.shape[1])):
		# 			errors.append(rms_angular_error_deg(DOA_batch_pred[i, pairing, ...].view(-1, 2),
		# 													   DOA_batch[i,...].view(-1, 2)))
		# 		batch_error += min(errors)
		# 	rmsae += batch_error / DOA_batch.shape[0]

		if return_rmsae:
			rmsae = 0
			for acoustic_scene in acoustic_scenes:
				rmsae += acoustic_scene.get_rmsae()
			return acoustic_scenes, rmsae / len(acoustic_scenes)
		else:
			return acoustic_scenes
