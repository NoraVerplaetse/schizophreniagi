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

from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, make_scorer, accuracy_score, matthews_corrcoef
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from statistics import mean, stdev
import socket
import sys, csv, copy, math, pickle, os, random, time, gc
import numpy as np


def main(args):
        """  main predictive function for the additive and non-linear baselines"""

        count=0
        X = np.load("SNPvector.npy")
        y = np.load("y.npy")

        model = SGDClassifier(loss='log_loss', penalty='l2', alpha=10000000)
        
        CV_SETS=3

        auctest=[]
        auctrain=[]

        run=0
        for i in range(10):
                cv=StratifiedKFold(CV_SETS, shuffle=True, random_state=i)
                auc_train=0
                auc_test=0
                
                cvrun=0
                for trainidx, testidx in cv.split(X,y):
                        Xtrain, ytrain = X[trainidx], y[trainidx]
                        print(Xtrain.shape, ytrain.shape)

                        model.fit(Xtrain, ytrain)
                        print('model fitted')
                        predtrain=model.predict_proba(Xtrain)[:,1]
                        del Xtrain
                        gc.collect()

                        Xtest, ytest = X[testidx], y[testidx]
                        predtest=model.predict_proba(Xtest)[:,1]
                        del Xtest
                        
                        auc_train += roc_auc_score(ytrain, predtrain)
                        auc_test += roc_auc_score(ytest, predtest)

                        cvrun+=1

                auctrain.append(auc_train/CV_SETS)
                auctest.append(auc_test/CV_SETS)
                           
if __name__ == "__main__":
        main(sys.argv)