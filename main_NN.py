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
import sources.utils as U
import sources.BioNets as BN
import sources.models as nn
import sources.wrappers as wr
from statistics import mean, stdev
from numpy import mean
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import numpy as np

def main(args):
	"""  main predictive function for the different neural net architectures
	Args:       gpu_devicename (str) 
	weight_decay (float) 
				dropout (float) 
				penalty (str): 'l1'or'l2'
				batch_size (int) epochs (int) 
	"""
	devicename = args[1]
	weight_decay=float(args[2])
	dropout=float(args[3])
	pen=args[4]
	batch_size=int(args[5])
	epochs=int(args[6])
	mode = args[7] if len(args) >7 else ""
	
	if mode == "toy":
		PATH= "toyExample/"

		X = np.load(PATH + "gene_centricTOY.npy")
		y = np.load(PATH + "yTOY.npy")
		globalNet = pickle.load(open(PATH + "globalNet.pickle", "rb"))

	else:

		# Gene-centric inputs
		path_to_dataX = PATH + "featvect.npy"
	
		# Mutation list inputs
		#path_to_dataX = PATH + "mutationlist.npy"
		#lengths = np.load(PATH + "mutationlist_lengths.npy")
	
		# Hilbert curve inputs
		#path_to_dataX = PATH + "hilbertcurves_zygosity.npy"
	
		path_to_datay = PATH + "y.npy"
		t1=time.time()
		X = np.load(path_to_dataX)
		y = np.load(path_to_datay)
		t2=time.time()
		print("Data loaded in ", t2-t1)
		
		genesdict=pickle.load(open(PATH + "genesdictRefGeneAnnovarComplete_withoutzerogenes", "rb"))
		genes=list(genesdict.values())

		#### CREATE SPARSE BIONET #####
		edges_genepw=list(pickle.load(open(PATH+ "hierData/kegg/edges_genepw.pickle", "rb")))
		for j in edges_genepw:
			if j[0] not in genes:
				edges_genepw.remove(j)
		print('after deleting edges not in genes', len(edges_genepw))
		pwlist = pickle.load(open("keggpwlist.pickle", "rb"))
		net_genepw=BN.BiologicalNetwork(edges_genepw, genes, pwlist)
		globalNet=BN.BiologicalHierarchy([net_genepw])
		print("BiologicalHierarchy built")

	device = t.device(devicename)
	hyperparameters={'device': t.device(devicename), 'epochs': epochs, 'learning_rate': 1e-3, 'weight_decay': weight_decay,'dropout': dropout, 'batch_size':batch_size}
	print("Running on ",hyperparameters['device'])
	print("Epochs: ", hyperparameters['epochs'], "Weight decay: ", hyperparameters['weight_decay'], "penalty: ", pen,  "Drop out: ", str(dropout), "batch size: ", hyperparameters['batch_size'])


	CV_SETS=3
	auctest=[]
	auctrain=[]
	run=0

	for i in range(1):
		cv=StratifiedKFold(CV_SETS, shuffle=True, random_state=i)
		cvrun=0
		auc_train=0
		auc_test=0

		for trainidx, testidx in cv.split(X, y): 
			print('run ' + str(run) + ' cv '+str(cvrun))
			Xtrainset, ytrainset = X[trainidx], y[trainidx]

			Xscaler= U.StructuredScalerGPU(devicename)
			Xtrainset = Xscaler.fit_transform(Xtrainset)
			print("shapes of Xtrain and ytrain ", Xtrainset.shape, ytrainset.shape)
			
			# Gene-centric models
			model=nn.NNbiosparse_GenePathway(X.shape[2], globalNet, dropout)
			#model=nn.NNlogreg(X.shape[2], X.shape[1], dropout)
			#model=nn.NNsmalldense(X.shape[2], X.shape[1], dropout)
			#model=nn.NNlargedense(X.shape[2], X.shape[1], dropout)
			#model=nn.NNdo(X.shape[2], X.shape[1], dropout)
			
			# Mutation list models
			#model=nn.GCN_mutlist((X.shape[2], 1, dropout)
			#model=nn.TNN_mutlist((X.shape[2], 1, dropout)
			
			# Hilbert Curve convolutional models
			#inputSpatialSize = (2**16,2**16)
			#model=nn.HCspcon1(1,1,inputSpatialSize, dropout=dropout)
			#model=nn.HCspcon2(1,1,inputSpatialSize, dropout=dropout)

			model.to(hyperparameters['device'])
			print("model built")

			# Gene-centric wrapper
			wrapper = wr.NNwrapperGC(model)
			
			# Mutation list wrapper
			#wrapper = wr.NNwrapperMutlist(model)

			# Hilbert Curve wrapper
			#wrapper = wr.NNwrapperHC(model)
			
			wrapper.fit(Xtrainset, ytrainset, hyperparameters['device'], epochs=hyperparameters['epochs'], batch_size=hyperparameters['batch_size'] , weight_decay = hyperparameters['weight_decay'], learning_rate=hyperparameters['learning_rate'], penalty=pen)
            
			predtrain=wrapper.predict(Xtrainset, hyperparameters['device'], batch_size=hyperparameters['batch_size'])
			
			auc_train += roc_auc_score(ytrainset, predtrain)
			del Xtrainset, ytrainset
			gc.collect()

			Xtest, ytest = X[testidx], y[testidx]
			Xtest = Xscaler.transform(Xtest)
			print("shapes of Xtest and ytest ", Xtest.shape, ytest.shape)
			predtest=wrapper.predict(Xtest, hyperparameters['device'], batch_size=hyperparameters['batch_size'])
			
			auc_test += roc_auc_score(ytest, predtest)

			del Xtest, ytest
			gc.collect()
			cvrun+=1

		auctrain.append(auc_train/CV_SETS)
		auctest.append(auc_test/CV_SETS)            

if __name__ == "__main__":
	main(sys.argv)
