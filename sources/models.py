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

import sklearn
import sys, csv, copy, math, pickle, os, random, time, socket, gc
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import torch as t
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, accuracy_score, average_precision_score
from statistics import mean, stdev

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import spconv.pytorch as spconv

### GENE-CENTRIC MODELS 
class NNlogreg(t.nn.Module):
		"""
		Class containing the NNlogreg architecture that consist of the shared gene module and a layer directly connection the gene nodes to the output.
		"""
	def __init__(self, genesize, numGenes, dropout):
		super(NNlogreg, self).__init__()
		self.genesize=genesize
		self.numGenes=numGenes
		self.dropout=dropout

		self.sharedgenemodule=t.nn.Sequential(t.nn.Dropout(self.dropout),t.nn.Linear(self.genesize, 50), t.nn.LayerNorm(50), t.nn.Tanh(), t.nn.Linear(50,1))
		self.output=t.nn.Sequential(t.nn.Dropout(self.dropout),t.nn.Linear(self.numGenes, 1))

		self.apply(init_weights)

	def forward(self, x):
		o=self.sharedgenemodule(x)
		o=self.output(o.squeeze())

		return o

class NNsmalldense(t.nn.Module):
		"""
		Class containing the NNsmalldense architecture that consist of the shared gene module and a fully connected hidden layer between the gene nodes and the output.
		"""
	def __init__(self, genesize, numGenes, dropout):
		super(NNsmalldense, self).__init__()
		self.genesize=genesize
		self.numGenes=numGenes
		self.dropout=dropout

		self.sharedgenemodule=t.nn.Sequential(t.nn.Dropout(self.dropout),t.nn.Linear(self.genesize, 50), t.nn.LayerNorm(50), t.nn.Tanh(), t.nn.Linear(50,1))
		self.hidden=t.nn.Sequential(t.nn.Dropout(self.dropout),t.nn.Linear(self.numGenes,5), t.nn.LayerNorm(5), t.nn.Tanh())
		self.output=t.nn.Sequential(t.nn.Dropout(self.dropout),t.nn.Linear(5, 1))

		self.apply(init_weights)

	def forward(self, x):
		o=self.sharedgenemodule(x)
		o=self.hidden(o.squeeze())
		o=self.output(o)

		return o
		
class NNlargedense(t.nn.Module):
		"""
		Class containing the NNlargedense architecture that consist of the shared gene module and a fully connected hidden layer between the gene nodes and the output.
		"""
	def __init__(self, genesize, numGenes, dropout):
		super(NNlargedense, self).__init__()
		self.genesize=genesize
		self.numGenes=numGenes
		self.dropout=dropout

		self.sharedgenemodule=t.nn.Sequential(t.nn.Dropout(self.dropout),t.nn.Linear(self.genesize, 50), t.nn.LayerNorm(50), t.nn.Tanh(), t.nn.Linear(50,1))
		self.hidden=t.nn.Sequential(t.nn.Dropout(self.dropout),t.nn.Linear(self.numGenes,50), t.nn.LayerNorm(50), t.nn.Tanh())
		self.output=t.nn.Sequential(t.nn.Dropout(self.dropout),t.nn.Linear(50, 1))

		self.apply(init_weights)

	def forward(self, x):
		o=self.sharedgenemodule(x)
		o=self.hidden(o.squeeze())
		o=self.output(o)

		return o

class NNbiosparse_GenePathway(t.nn.Module):
		"""
		Class containing the NNbiosparse_GenePathway architecture that consist of the shared gene module and a sparsified hidden layer, representing biological pathways between the gene nodes and the output.
		"""
	def __init__(self, genesize, bioNetworks, dropout):
		super(NNbiosparse_GenePathway, self).__init__()
		self.genesize=genesize
		self.dropout=dropout

		self.sharedgenemodule=t.nn.Sequential(t.nn.Dropout(self.dropout),t.nn.Linear(self.genesize, 50), t.nn.LayerNorm(50), t.nn.Tanh(), t.nn.Linear(50,1))
		self.sparseGenepw=t.nn.Sequential(t.nn.Dropout(self.dropout),SparseLinear(bioNetworks[0]), t.nn.LayerNorm(bioNetworks[0].size()[1]), t.nn.Tanh())
		self.output=t.nn.Sequential(t.nn.Dropout(self.dropout),t.nn.Linear(bioNetworks[0].size()[1], 1))

		self.apply(init_weights)

	def forward(self, x):
		o=self.sharedgenemodule(x)
		o=self.sparseGenepw(t.squeeze(o,2))
		o=self.output(o)

		return o
		
class NNdo(t.nn.Module):
		"""
		Class containing the NNdropout architecture that consist of the shared gene module and a fully connected hidden layer between the gene nodes and the output, with a heavy dropout.
		"""
	def __init__(self, genesize, numGenes, dropout):
		super(NNdo, self).__init__()
		self.genesize=genesize
		self.numGenes=numGenes
		self.dropout=dropout

		self.sharedgenemodule=t.nn.Sequential(t.nn.Dropout(self.dropout),t.nn.Linear(self.genesize, 50), t.nn.LayerNorm(50), t.nn.Tanh(), t.nn.Linear(50,1))
		self.hidden=t.nn.Sequential(t.nn.Dropout(0.95),t.nn.Linear(self.numGenes,280), t.nn.LayerNorm(280), t.nn.Tanh())
		self.output=t.nn.Sequential(t.nn.Dropout(self.dropout),t.nn.Linear(280, 1))

		self.apply(init_weights)

	def forward(self, x):
		o=self.sharedgenemodule(x)
		o=self.hidden(o.squeeze())
		o=self.output(o)

		return o        

class SparseLinear(t.nn.Module):
		"""Class representing a computational efficient format of the sparsified layer.

		Args:
				n_in (int): Number of input features.
				n_out (int): Number of neurons.
				row_idx (np.ndarray): row indices of the COO matrix.
						Its length defines the number of parameters in the matrix.
				col_idx (np.ndarray): column indices of the COO matrix.
						Must be the same size as `row_idx`.
						bias (bool): Whether to include a (dense) bias term.
		"""
		def __init__(self, bioNet, bias=True):
				t.nn.Module.__init__(self)
				self.edges, self.size = bioNet.getDataForCOO()
				self.n_in = len(bioNet.nodes1)
				self.n_out = len(bioNet.nodes2)
				self.n_elements = self.edges.shape[1]
				row_idx=self.edges[0]
				col_idx=self.edges[1]
		# Sort indices by column, so the indexed elements associated to
		# the same output neuron are adjacent. Each set of adjacent elements
		# associated to the same output neuron defines a separate bin.
				row_idx = np.asarray(row_idx)
				col_idx = np.asarray(col_idx)

				assert (np.min(row_idx) >= 0) and (np.max(row_idx) < self.n_in)
				assert (np.min(col_idx) >= 0) and (np.max(col_idx) < self.n_out)
				idx = np.argsort(col_idx)
				col_idx, row_idx = col_idx[idx], row_idx[idx]
				self.col_idx = t.LongTensor(col_idx)
				self.row_idx = t.LongTensor(row_idx)
		# TODO: remove redundant index pairs, if any

		# Find the start and end indices of each bin.
				starts, ends, bounds_mask = SparseLinear._find_bins_bounds(col_idx, self.n_out)
				self.starts = t.LongTensor(starts)
				self.ends = t.LongTensor(ends)
				self.bounds_mask = t.FloatTensor(bounds_mask)
				assert len(self.starts) == self.n_out
				assert len(self.ends) == self.n_out

		# Sparse parameters, stored in a dense tensor.
				self.weight = t.nn.Parameter(t.FloatTensor(np.empty(self.n_elements, dtype=float)))

		# Xavier initialization
				#t.nn.init.normal_(self.weight, 0, 2. / (self.n_in + self.n_out))
				#t.nn.init.normal_(self.weight, 0, 0.1)
				#t.nn.init.xavier_uniform_(self.weight.view(1,-1))
				# Dense bias term (if needed)
				if bias:
						#self.bias = t.nn.Parameter(t.zeros((1, self.n_out)))
						self.bias = t.nn.Parameter(t.ones((1, self.n_out))*0.1)

				else:
						self.bias = None

		def forward(self, X):
		# Sparse matrix multiplication
				X = X[:, self.row_idx] * self.weight.view(1, -1)
				X = t.cumsum(X, 1)  # Voodoo magic
				X = X[:, self.ends] - self.bounds_mask.to(self.weight.device) * X[:, self.starts]

		# Dense bias addition
				if self.bias is not None:
						X = X + self.bias
				return X

		@staticmethod
		def _find_bins_bounds(idx, n_out):
				"""Find the start and end of each bin.
				The sum of a bin is computed as the overall cumulative sum at the
				last position of the bin, minus the overall cumulative sum at the
				position that precedes the start of the bin.

				For the first bin, which is not preceded by anything,
				the start index is replaced by 0. Also, to avoid subtracting the
				first element from the sum, the `bounds_mask` array is used to
				cancel out the related term:

				.. math::
						Y_{ik} = C_{i e_k} - b_k C_{i s_k}
						C_l = \sum\limits_{(p, q, r) \in L} X_{ip} w_r

				where `C` is the matrix of cumulative sums, `s` is the vector `starts`,
				`e` is the vector `ends`, `b` is the vector `bounds_mask`,
				`w` is the parameter vector, and `I` is the set of allowed indices
				in the sparse matrix.

				Args:
						idx (np.ndarray): Output neuron indices, which are the column
								indices of the COO matrix.
						n_out (int): The number of output neurons.

				Returns:
						starts (np.ndarray): Indices that precede the start of each bin.
						ends (np.ndarray): Last index of each bin. Array that has the
								same length as `starts`.
						bounds_mask (np.ndarray): mask which has the same length as `starts`.
				"""
				if n_out == 0:
						return np.asarray([]), np.asarray([]), np.asarray([])
				idx = np.asarray(idx, dtype=int)

		# Locations of index changes (contiguity break)
				changes = np.where(np.diff(idx, prepend=-2, append=-1) != 0)[0]
				changes = np.maximum(0, changes - 1)

		# Starts and ends are set to 0 by default, so the bin sum will
		# be zero if no index is found for that bin.
				starts = np.zeros(n_out, dtype=int)
				ends = np.zeros(n_out, dtype=int)

		# For bins containing at least one index, store the start and
		# end locations of those indices.
				xs = idx[changes][1:]
				starts[xs] = changes[:-1]
				ends[xs] = changes[1:]

		# Subtract the cumulative sum up to the first element of the bin,
		# except for the first bin (where cumulative sum necessarily
		# starts at 0). Subtraction is performed when
		# `bounds_mask[k]` is True.
				bounds_mask = np.ones(n_out)
				bounds_mask[0] = 0

				return starts, ends, bounds_mask

### MUTATION LIST MODELS

def masking(X, X_len):
		"""
		Function for masking required given the variable input size of the mutation list (depending on the number of exonic variants in the sample).
		"""
		maxlen = X.size(1)
		mask = t.arange(maxlen).to(X.device)[None, :] < X_len[:, None]
		mask=t.unsqueeze(mask,2)
		mask=mask.expand(-1,-1,X.shape[2])

		return ~mask

class GCN_mutlist(t.nn.Module): 
		"""
		Class containing the GCN_mutlist architecture which consists of two graph NN layers implementing a hierarchical graph betweeen variants and genes, and genes and pathways respectively.
		"""
		def __init__(self, numGenes, numOut, dropout, name = "NN", vartypeembsize =5, geneembsize=50):
				super(GCN_mutlist, self).__init__()
				os.system("mkdir -p models")
				self.name = name
				self.numOut = numOut
				self.numGenes = numGenes
				self.dropout = dropout

				VARTYPESIZE = vartypeembsize
				GENESIZE = geneembsize
				self.vartypeEmb = t.nn.Embedding(11, VARTYPESIZE)

				self.processVartype =  t.nn.Sequential(t.nn.Linear(VARTYPESIZE, 20), t.nn.LayerNorm(20), t.nn.Tanh(), t.nn.Linear(20,5), t.nn.LayerNorm(5), t.nn.Tanh(), t.nn.Dropout(self.dropout))
				self.processGene = t.nn.Sequential(t.nn.Linear(4, 20), t.nn.LayerNorm(20), t.nn.Tanh(), t.nn.Linear(20,5), t.nn.LayerNorm(5), t.nn.Tanh(), t.nn.Dropout(self.dropout))
				self.processMuts = t.nn.Sequential(t.nn.Linear(4, 20), t.nn.LayerNorm(20), t.nn.Tanh(), t.nn.Linear(20,10), t.nn.LayerNorm(10), t.nn.Tanh(), t.nn.Dropout(self.dropout))

				self.adjcoor_genepw = t.tensor(np.transpose(np.load("adjcoor_genepw.npy")))
				self.numPathways = 280

				self.preFinal = t.nn.Sequential(t.nn.Linear(20, 25), t.nn.LayerNorm(25), t.nn.Tanh(), t.nn.Linear(25,1))
				self.final = t.nn.Sequential(t.nn.Dropout(self.dropout), t.nn.LayerNorm(self.numGenes), t.nn.Tanh(), t.nn.Linear(self.numGenes,self.numPathways), t.nn.LayerNorm(self.numPathways), t.nn.Tanh(), t.nn.Linear(self.numPathways, numOut))                

				self.apply(init_weights)

		def forward(self, x, xlens):
				batch_size = x.size()[0]
				genesIndices = x[:,:,0].to(t.long).reshape(-1) 
				ev = self.vartypeEmb(x[:,:,1].to(t.int))
				eg = self.processGene(x[:,:,2:6])
				ev = self.processVartype(ev)
				x = self.processMuts(x[:,:,6:])
				x = t.cat([eg, ev, x], dim=2)
				
				mask=masking(x, xlens)
				x=x.masked_fill(mask,0)

				mutIndices = t.arange(15000).to(x.device).expand(batch_size,-1).reshape(-1) 
				batch = t.arange(batch_size).to(x.device).unsqueeze(1).expand(-1,15000).reshape(-1)
				i = t.cat([batch.unsqueeze(0), genesIndices.unsqueeze(0), mutIndices.unsqueeze(0)], dim=0)
				adj_mutgene = t.sparse_coo_tensor(i, t.ones(i.size()[1]), [batch_size, self.numGenes, 15000], device=x.device) 
				x = t.bmm(adj_mutgene, x) 

				batch = t.arange(batch_size).to(x.device).unsqueeze(1).expand(-1,self.adjcoor_genepw.shape[1]).reshape(-1)
				pwIndices=self.adjcoor_genepw[0,:].repeat(batch_size).to(x.device)
				geneIndices=self.adjcoor_genepw[1,:].repeat(batch_size).to(x.device)
				i = t.cat([batch.unsqueeze(0), pwIndices.unsqueeze(0), geneIndices.unsqueeze(0)], dim=0)
				adj_genepw = t.sparse_coo_tensor(i, t.ones(i.size()[1]), [batch_size, self.numPathways, self.numGenes], device=x.device)
				x = t.bmm(adj_genepw, x)
				
				x=self.preFinal(x).squeeze()
				x = self.final(x)
				
				return x

class TNN_mutlist(t.nn.Module): 
		"""
		Class containing the TNN_mutlist architecture which a biologically sparsified transformer structure on top of the mutation list encoding.
		"""
		def __init__(self, numGenes, numOut, dropout, name = "NN", vartypeembsize =5, geneembsize=50, attSize=10):
				super(TNN_mutlist, self).__init__()
				os.system("mkdir -p models")
				self.name = name
				self.numOut = numOut
				self.numGenes = numGenes
				self.dropout = dropout
				self.vartypeend = 5
				self.processgeneend = 5
				self.processmutsend = 10

				VARTYPESIZE = vartypeembsize
				GENESIZE = geneembsize
				self.vartypeEmb = t.nn.Embedding(10, VARTYPESIZE)
				self.processVartype =  t.nn.Sequential(t.nn.Linear(VARTYPESIZE, 20), t.nn.LayerNorm(20), t.nn.Tanh(), t.nn.Linear(20,self.vartypeend), t.nn.LayerNorm(self.vartypeend), t.nn.Tanh(), t.nn.Dropout(self.dropout))
				self.processGene = t.nn.Sequential(t.nn.Linear(4, 20), t.nn.LayerNorm(20), t.nn.Tanh(), t.nn.Linear(20,self.processgeneend), t.nn.LayerNorm(self.processgeneend), t.nn.Tanh(), t.nn.Dropout(self.dropout))
				self.processMuts = t.nn.Sequential(t.nn.Linear(4, 20), t.nn.LayerNorm(20), t.nn.Tanh(), t.nn.Linear(20,self.processmutsend), t.nn.LayerNorm(self.processmutsend), t.nn.Tanh(), t.nn.Dropout(self.dropout))
				self.embSize = self.processgeneend + self.processmutsend + self.vartypeend
				self.weight = t.nn.Parameter(t.randn((self.numGenes, self.embSize)))

				self.toQuery = t.nn.Sequential(t.nn.Linear(self.embSize, attSize), t.nn.Tanh())
				self.toKey = t.nn.Sequential(t.nn.Linear(self.embSize, attSize), t.nn.Tanh())
				self.toValue = t.nn.Sequential(t.nn.Linear(self.embSize, attSize), t.nn.Tanh())

				self.s = t.nn.Softmax(dim=2)
				self.sqrt = math.sqrt(attSize)
				self.attSize = attSize
				self.afterGene = t.nn.Sequential(t.nn.Linear(self.attSize, 25), t.nn.LayerNorm(25), t.nn.Tanh(), t.nn.Linear(25,1))

				self.final = t.nn.Sequential(t.nn.Dropout(self.dropout), t.nn.LayerNorm(self.numGenes), t.nn.Tanh(), t.nn.Linear(self.numGenes,numOut)) 
				self.apply(init_weights)

		def forward(self, x, xlens):
				batch_size = x.size()[0]
				genesIndices = x[:,:,0].to(t.long).reshape(-1)

				ev = self.vartypeEmb(x[:,:,1].to(t.int))
				eg = self.processGene(x[:,:,2:6])
				ev = self.processVartype(ev)
				x = self.processMuts(x[:,:,6:])
				x = t.cat([eg, ev, x], dim=2)
				
				mask=masking(x, xlens)
				x=x.masked_fill(mask,0)

				mutIndices = t.arange(15000).to(x.device).expand(batch_size,-1).reshape(-1)
				batch = t.arange(batch_size).to(x.device).unsqueeze(1).expand(-1,15000).reshape(-1)
				i = t.cat([batch.unsqueeze(0), genesIndices.unsqueeze(0), mutIndices.unsqueeze(0)], dim=0)

				adj_mutgene = t.sparse_coo_tensor(i, t.ones(i.size()[1]), [batch_size, self.numGenes, 15000], device=x.device) 
				adj_mutgene = adj_mutgene.to_dense().type(t.bool)

				q = self.toQuery(self.weight) 
				k = self.toKey(x)
				v = self.toValue(x)
				qk = t.matmul(q, k.transpose(1,2))/self.sqrt 

				adj_mutgene.to(self.weight.device)
				qk[~adj_mutgene] = -9e15 
				del adj_mutgene
				gc.collect()

				qk = self.s(qk)
				qk = t.matmul(qk,v)
				
				o = self.afterGene(qk).squeeze()
				out = self.final(o)
				
				return out


### HILBERT CURVE MODELS
class HCspconv1(t.nn.Module):
		"""
		Class containing the first tried sparse convolutional model that takes the one-channel hilbert curves as input.
		"""
		def __init__(self, featSize, numOut, size,name = "NN", dropout=0.0 ):
				super(HCspconv1, self).__init__()
				print(size)
				self.inputSpatialSize = size
				self.sparseModel = spconv.SparseSequential(spconv.SparseConv2d(1,1,8,8), t.nn.LeakyReLU(), spconv.SparseConv2d(1,4,8,8), t.nn.LayerNorm(4), t.nn.LeakyReLU(), spconv.SparseConv2d(4,16,3,2), t.nn.LayerNorm(16), t.nn.LeakyReLU(), spconv.SparseConv2d(16,32,3,2), t.nn.LayerNorm(32), t.nn.LeakyReLU(), spconv.SparseConv2d(32,64,3,2), t.nn.LayerNorm(64), t.nn.LeakyReLU(), spconv.SparseConv2d(64,128,3,2), t.nn.LayerNorm(128), t.nn.LeakyReLU(), spconv.SparseConv2d(128,128,3,2), t.nn.LayerNorm(128), t.nn.LeakyReLU(), spconv.SparseConv2d(128,128,3,2), t.nn.LayerNorm(128), t.nn.LeakyReLU()) 
				self.final = t.nn.Sequential(t.nn.Dropout(dropout), t.nn.Linear(15*15*128, 50), t.nn.LeakyReLU(), t.nn.Linear(50, numOut))
				self.apply(init_weights)

		def forward(self, x, batchSize=1):
				x = self.sparseModel(x)
				x = x.dense()
				x = self.final(x.view(batchSize, -1))
				return x
								
class HCspconv2(t.nn.Module):
		"""
		Class containing the third tried sparse convolutional model that takes the one-channel hilbert curves as input.
		"""
		def __init__(self, featSize, numOut, size,name = "NN", dropout=0.0 ):
				super(HCspconv2, self).__init__()
				print(size)
				self.inputSpatialSize = size
				self.sparseModel = spconv.SparseSequential(spconv.SparseConv2d(1,1,11,4), t.nn.LeakyReLU(), spconv.SparseConv2d(1,4,3,2), t.nn.LayerNorm(4), t.nn.LeakyReLU(), spconv.SparseConv2d(4,16,3,2), t.nn.LayerNorm(16), t.nn.LeakyReLU(), spconv.SparseConv2d(16,32,3,2), t.nn.LayerNorm(32), t.nn.LeakyReLU(), spconv.SparseConv2d(32,64,3,2), t.nn.LayerNorm(64), t.nn.LeakyReLU(), spconv.SparseConv2d(64,128,3,2), t.nn.LayerNorm(128), t.nn.LeakyReLU(), spconv.SparseConv2d(128,128,3,2), t.nn.LayerNorm(128), t.nn.LeakyReLU(), spconv.SparseConv2d(128,128,3,2), t.nn.LayerNorm(128), t.nn.LeakyReLU(), spconv.SparseConv2d(128,128,3,2), t.nn.LayerNorm(128), t.nn.LeakyReLU(), spconv.SparseConv2d(128,128,3,2), t.nn.LayerNorm(128), t.nn.LeakyReLU(), spconv.SparseConv2d(128,128,3,2), t.nn.LayerNorm(128), t.nn.LeakyReLU())
				self.final = t.nn.Sequential(t.nn.Dropout(dropout), t.nn.Linear(14*14*128, numOut))#, t.nn.LeakyReLU(), t.nn.Linear(50, numOut))
				self.apply(init_weights)

		def forward(self, x, batchSize):
				x = self.sparseModel(x)
				x = x.dense()
				x = self.final(x.view(batchSize, -1))
				return x


def init_weights( m):
		"""
		Function for weight initialization of the neural network depending on the layer type.
		"""
	if isinstance(m, t.nn.Conv1d) or isinstance(m, t.nn.Linear) or isinstance(m, spconv.SparseConv2d) or isinstance(m, t.nn.Conv2d):
		print ("Initializing weights...", m.__class__.__name__)
		#t.nn.init.normal(m.weight, 0, 0.01)
		t.nn.init.xavier_uniform_(m.weight)
		m.bias.data.fill_(0.1)
	elif isinstance(m, SparseLinear):
		print ("Initializing weights...", m.__class__.__name__)
		t.nn.init.xavier_uniform_(m.weight.view(1,-1))
		m.bias.data.fill_(0.1)
	elif isinstance(m, t.nn.Embedding):
		print ("Initializing weights...", m.__class__.__name__)
		t.nn.init.xavier_uniform_(m.weight)
