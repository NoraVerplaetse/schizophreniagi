#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  untitled.py
#  
#  Copyright 2023 Daniele Raimondi, Nora Verplaetse
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.

import torch as t
import sys, csv, copy, math, pickle, os, random, time, gc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import utils as U
import BioNets as BN
import models as nn
from statistics import mean, stdev
from numpy import mean
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.sparse import coo_matrix
import spconv

class myDatasetGC(Dataset):
	""" 
	Specific application of Pytorch Dataset class for use in gene-centric models.
	"""
	def __init__(self, X, y, pred=False):
		self.X=t.tensor(X, dtype=t.float32)
		self.y=t.tensor(y, dtype=t.float32)
		self.pred=pred
		
	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		if self.pred == False:
			return self.X[idx], self.y[idx]
		else:
			return self.X[idx]
			
class myDatasetMutList(Dataset):
	""" 
	Specific application of Pytorch Dataset class for use in mutation list models.
	"""
	def __init__(self, X, Y, lens, pred=False):
		self.X = t.from_numpy(X)
		self.Y = t.from_numpy(Y)
		self.lens = t.from_numpy(lens)
		self.pred=pred

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		if self.pred == False:
			return self.X[idx], self.Y[idx], self.lens[idx]
		else:
			return self.X[idx], self.lens[idx]
		
class myDatasetHC(Dataset):
	""" 
	Specific application of Pytorch Dataset class for use in Hilbert curve models.
	"""
	def __init__(self, X, Y, pred=False):
		self.X = X
		self.Y = t.tensor(Y, dtype=t.float32)
		self.pred=pred
		
	def __len__(self):
		return len(self.X)
		
	def __getitem__(self, idx):
		if self.pred == False:
			if type(self.X[0]) == coo_matrix:
				return np.stack([self.X[idx].row, self.X[idx].col], axis=1), self.X[idx].data, self.Y[idx]
			elif type(self.X[0]) == t.Tensor:
				return self.X[idx].coalesce().values(), self.X[idx].coalesce().indices(), self.Y[idx]
			else:
				return np.stack([self.X[idx].row, self.X[idx].col], axis=1), self.X[idx].data, self.Y[idx]
		else:
			if type(self.X[0]) == coo_matrix:
				return np.stack([self.X[idx].row, self.X[idx].col], axis=1), self.X[idx].data
				#return np.stack([self.X[idx].row, self.X[idx].col], axis=1),np.ones((len(self.X[idx].row),1))
			elif type(self.X[0]) == t.Tensor:
				return self.X[idx].coalesce().values(), self.X[idx].coalesce().indices()
			else:
				return np.stack([self.X[idx].row, self.X[idx].col], axis=1), self.X[idx].data                

def SparseConvNetCollate(batch):
	""" 
	Collate function adapted to use sparse convolutions in Hilbert curve models.
	"""
	z = []
	x = []
	for i, item in enumerate(batch):
		x.append(item[0])
		z.append(t.ones((item[0].shape[0], 1))*i)
	batchSize = len(batch)
	x = t.tensor(np.concatenate(x, axis=0))
	z = t.cat(z, dim=0)
	x = t.cat([z, x], dim=1)
	data = t.tensor(np.concatenate([item[1] for item in batch], axis=0))
	if len(data.size())<2:
		data = data.unsqueeze(1)
	if len(batch[0]) == 3:
		y = t.stack([item[2] for item in batch], dim=0)
		return x.to(t.int), data.to(t.float), y, batchSize
	if len(batch[0]) == 2:
		return x.to(t.int), data.to(t.float), batchSize

class NNwrapperGC():
	""" 
	Wrapper class containing functionality for neural net training (fit function) and prediction (predict function) for gene-centric models.
	"""
	def __init__(self, model):
		if type(model) == str:
			self.model = t.load(model)
		else:
			self.model = model

	def fit(self, originalX, originalY, device, epochs = 1, batch_size=20, weight_decay = 0.001, learning_rate = 1e-3, penalty='l2'):
		########DATASET###########
		t1=time.time()
		dataset = myDatasetGC(originalX, originalY)
		t2=time.time()
		print("Dataset created in %.2fs" % (t2-t1))

		#######MODEL##############
		self.model.train()
		print("Start training")
		print(" Number of epochs:", epochs, " Batch size:", batch_size, " weight decay:", weight_decay, " learing rate:", learning_rate)

		########LOSS FUNCTION######
		lossfn = t.nn.BCEWithLogitsLoss()
		print(lossfn)
		
		########OPTIMIZER##########
		self.learning_rate = learning_rate

		parameters = list(self.model.parameters())

		p=[]
		for i in parameters:
			p+=list(i.data.cpu().numpy().flat)
		print("Number of parameters=", len(p))
		del p
		
		optimizer =t.optim.Adam(parameters, lr=self.learning_rate, weight_decay=0)
		scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=10, verbose=True)
			  
		t1=time.time()
		loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, sampler=None, num_workers=0)      
		t2=time.time()
		print("Dataloader created in %.2fs" % (t2-t1))
		
		e = 1
		minLoss = 1000000000
		while e <= epochs+1:
			errTot = 0

			i = 1
			start = time.time()

			for sample in loader:
				optimizer.zero_grad()
				
				x, y = sample
				x=x.to(device)
				y=y.to(device)

				yp = self.model(x)

				loss = lossfn(yp.squeeze().float(), y.float())

				if penalty=="l2":
					norm = sum(p.pow(2.0).sum() for p in self.model.parameters())
				elif penalty=='l1':
					norm = sum(p.abs().sum() for p in self.model.parameters())
				else:
					print("No allowed penalty given")

				loss = loss + weight_decay*norm
				loss.backward()
				optimizer.step()
				errTot += loss.detach().item()

			end = time.time()
			scheduler.step(errTot)
			print(" epoch %d, ERRORTOT: %f (%fs)" % (e, errTot, end-start))
			e += 1

	def predict(self, X, device, batch_size=None):
		self.model.eval()
		dataset = myDatasetGC(X,[], pred=True)
		if batch_size == None:
			batch_size = len(X)
		loader = DataLoader(dataset, batch_size, shuffle=False, sampler=None, num_workers=0)
		
		preds1=[]
		i=0
		for sample in loader:
			i+=1
			x= sample
			x= t.tensor(x, dtype=t.float, device=device)
			y_pred = self.model(x)        
			y_pred=t.sigmoid(y_pred)

			if y_pred.data.squeeze().shape == t.Size([]):
				preds1+=[y_pred.data.squeeze().item()]
			else:
				preds1+=y_pred.data.squeeze().tolist()
		
		return np.array(preds1)
		
class NNwrapperMutList():
	""" 
	Wrapper class containing functionality for neural net training (fit function) and prediction (predict function) for mutation list models.
	"""
	def __init__(self, model, device):
		if type(model) == str:
			self.model = t.load(model)
		else:
			self.model = model
		self.device = device

	def fit(self, originalX, originalY, lens, IGNORE_INDEX, epochs = 50, batch_size=1, lossWeights = None, weight_decay = 1e-2, learning_rate = 1e-3):
		########DATASET###########
		t1 = time.time()
		dataset = myDatasetMutList(originalX, originalY, lens)
		t2 = time.time()
		print ("Dataset created in %.2fs" % (t2 - t1))
		
		#######MODEL##############
		self.model.train()
		print ("Start training")
		
		########LOSS FUNCTION######
		lossfn=t.nn.BCEWithLogitsLoss()
		parameters =list(self.model.parameters())
		p = []
		for i in parameters:
			p+= list(i.data.cpu().numpy().flat)
		print ('Number of parameters=',len(p))
		del p
		self.model.train()
		print ("Training mode: ", self.model.training)

		optimizer = t.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
		scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2, verbose=True, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

		########DATALOADER#########
		t1 = time.time()
		loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, sampler=None, num_workers=0)
		t2 = time.time()
		print( "Dataloader created in %.2fs" % (t2 - t1))
		print ("Start epochs")
		e = 1
		minLoss = 1000000000

		while e < epochs:
			errTot = 0
			i = 1
			start = time.time()
			t1 = time.time()
			for sample in loader:
				optimizer.zero_grad()
				x, y, xl = sample
				x = x.to(self.device).to(t.float)
				xl = xl.to(self.device)
				y = y.to(self.device)
				yp = self.model.forward(x, xl)
				loss = lossfn(yp.squeeze().float(), y.float())
				loss.backward()
				optimizer.step()
				errTot += loss.data
				i+=batch_size
				perc = (100*i/float(len(dataset))       )

			end = time.time()
			scheduler.step(errTot)
			e += 1
					
	def predict(self, X, lens, batch_size = 5):
		self.model.eval()
		print ("Predicting...")
		dataset = myDatasetMutList(X, [], lens, pred=True)
		loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=None, num_workers=0)
		preds1 = []
		act = []
		for sample in loader:
			x, xl = sample
			xl = xl.to(self.device)
			x = x.to(t.float32).to(self.device)

			y_pred = self.model.forward(x, xl)
			y_pred=t.sigmoid(y_pred) 

			if y_pred.data.squeeze().shape == t.Size([]):
				preds1+= [y_pred.data.squeeze().item()]
			else:
				preds1+= y_pred.data.squeeze().tolist()
		return np.array(preds1)
	
class NNwrapperHC():
	""" 
	Wrapper class containing functionality for neural net training (fit function) and prediction (predict function) for Hilbert curve models.
	"""
	def __init__(self, model):
		if type(model) == str:
			self.model = t.load(model)
		else:
			self.model = model

	def fit(self, originalX, originalY, device, epochs = 50, batch_size=11, lossWeights = None, save_model_every=10, weight_decay = 1e-9, learning_rate = 1e-5, silent = False, IGNORE_INDEX = -9999, LOG=False):
		########DATASET###########
		t1 = time.time()
		dataset = myDatasetHC(originalX, originalY)
		t2 = time.time()
		print ("Dataset created in %.2fs" % (t2 - t1))
		
		#######MODEL##############
		self.model.train()
		print ("Start training")
		
		########LOSS FUNCTION######
		lossfn = t.nn.BCEWithLogitsLoss()

		########OPTIMIZER##########
		try:
			parameters =list(self.model.sparseModel.parameters())+ list(self.model.final.parameters())
		except AttributeError:
			parameters =list(self.model.parameters())
		p = []
		for i in parameters:
			p+= list(i.data.cpu().numpy().flat)
			#print(len(p))
		print ('Number of parameters=',len(p))
		del p
		self.model.train()
		print ("Training mode: ", self.model.training)

		optimizer = t.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
		scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=3, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

		########DATALOADER#########
		t1 = time.time()
		loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=None, num_workers=1, collate_fn = SparseConvNetCollate)

		t2 = time.time()
		print ("Dataloader created in %.2fs" % (t2 - t1))
		print ("Start epochs")
		e = 1
		minLoss = 1000000000
		while e < epochs+1:
			errTot = 0
			i = 1
			start = time.time()
			tload = 0
			tforw = 0
			tback = 0
			t1 = start
			for sample in loader:
				coord, features, y, batchSize = sample

				optimizer.zero_grad()
				x = spconv.SparseConvTensor(features.to(device), coord.to(device), self.model.inputSpatialSize, batchSize)

				y = y.to(device)
				tload += time.time() - t1
				t1 = time.time()
				yp = self.model.forward(x, batchSize)
				tforw += time.time() - t1
				t1 = time.time()
				loss = lossfn(yp.squeeze().float(), y.float())

				loss.backward()
				optimizer.step()
				tback += time.time() - t1
				errTot += loss.data
				i+=batch_size
				perc = (100*i/float(len(dataset))       )
				t1 = time.time()
			end = time.time()

			scheduler.step(errTot)
			
			e += 1

	def predict(self, X, device, batch_size = 11):
		self.model.eval()
		print ("Predicting...")
		dataset = myDatasetHC(X,[],pred=True)
		loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=None, num_workers=1, collate_fn=SparseConvNetCollate)
		preds1 = []
		act = []
		for sample in loader:
			coord, features, batchSize = sample
			x = spconv.SparseConvTensor(features.to(device), coord.to(device), self.model.inputSpatialSize, batchSize)
			y_pred = self.model.forward(x, batchSize)
			y_pred=t.sigmoid(y_pred)
		
			if y_pred.data.squeeze() == t.Size([]):
					preds1 += [y_pred.data.squeeze().tolist()]
			else:
					preds1 += y_pred.data.squeeze().tolist()

		return np.array(preds1)
			
