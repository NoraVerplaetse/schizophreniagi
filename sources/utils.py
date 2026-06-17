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

import numpy as np
import time
import torch as t

class StructuredScalerGPU:
	"""
	A class to standardize the gene-centric encoding on GPU. 
	The integer counts of the 16 types of variants will be standardize on all genes and all samples,
	to ensure that the global distribution of the occurrences of each type of variant on the entire dataset is a Gaussian with mean 0 and variance 1, 
	making it more suitable for the neural net optimization.
	"""
	def __init__(self, device, epsilon=1e-9):
		self.device = device
		self.epsilon = epsilon

	def fit_transform(self, X):
		t1 = time.time()
		if type(X) != t.tensor:
			X = t.tensor(X)
		X = X.to(self.device).to(t.float32)
		t2 = time.time()
		shape = X.size()

		X = X.view(shape[0]*shape[1], shape[2])
		self.mean = t.mean(X, axis=0)
		self.std = t.std(X, axis=0)
		self.std[self.std == 0] = self.epsilon
		X = (X - self.mean) / self.std
		X = X.view(shape[0], shape[1], shape[2])
		t2 = time.time()
		print("Transform done in %.3fs"% (t2-t1))
		return X.cpu().numpy()

	def transform(self, X):
		t1 = time.time()
		if type(X) != t.tensor:
				X = t.tensor(X)
		X = X.to(self.device)
		shape = X.size()
		X = X.view(shape[0]*shape[1], shape[2])

		X = (X - self.mean) / self.std
		X = X.view(shape[0], shape[1], shape[2])
		t2 = time.time()

		print("Transform done in %.3fs"% (t2-t1))
		return X.cpu().numpy()