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

from multipledispatch import dispatch
import pickle
import torch as t
import numpy as np

class LookupList(object):
"""
Class defining a new data structure combining list and dictionary functionalities.
"""
	def __init__(self, data=None):
		self.lookup = {}
		self.mylist = []
		self.counter = 0
		if data != None:
			for i in data:
				self.mylist.append(i)
				self.lookup[i] = self.counter
				self.counter += 1

	def index(self, key):
		return self.lookup[key]

	@dispatch(int)
	def __getitem__(self, key):
		return self.mylist[key]

	@dispatch(str)
	def __getitem__(self, key):
		return self.lookup[key]

	def append(self, v):
		self.mylist.append(v)
		self.lookup[v] = self.counter
		self.counter += 1

	def remove(self, v):
		if type(v) == list or type(v) == set:
			if len(v) == 0:
				return
			olen = len(self)
			for i in v:
				self.mylist.remove(i)
				del self.lookup[i]
			assert len(self) == olen - len(v)
			self.counter = 0
			for i in self.mylist:
				self.lookup[i] = self.counter
				self.counter += 1
			assert len(self.lookup) == len(self.mylist)
		else:
			self.remove([v])

	@dispatch(list)
	def __add__(self, l):
		for i in l:
			self.append(i)
		return self

	@dispatch(list)
	def extend(self, l):
		for i in l:
			self.append(i)

	def __iter__(self):
		return iter(self.mylist)

	def __len__(self):
		return len(self.mylist)

	def __str__(self):
		return ""+str(self.mylist)

class BiologicalNetwork(object):
"""
Class providing the needed functionalities to transform information on edges and nodes of a biological network into a BiologicalNetwork object that can be used to build biologically meaningful sparsified neural network layers.
"""
	@dispatch(list, list)
	def __init__(self, edges, nodes1):
		self.edges = edges
		self.nodes2 = set()
		for i in self.edges:
			self.nodes2.add(i[1])
		self.nodes1 = LookupList(nodes1)
		self.nodes2 = LookupList(self.nodes2)

	@dispatch(list, list, list)
	def __init__(self, edges, nodes1, nodes2):
		self.edges = edges
		self.nodes1 = LookupList(nodes1)
		self.nodes2 = LookupList(nodes2)

	@dispatch(list, LookupList)
	def __init__(self, edges, nodes1):
		self.edges = edges
		self.nodes2 = set()
		for i in self.edges:
			self.nodes2.add(i[1])
		self.nodes1 = nodes1
		self.nodes2 = LookupList(self.nodes2)

	def __str__(self):
		return "Nodes1: %d, nodes2: %d, edges: %d" % (len(self.nodes1), len(self.nodes2), len(self.edges))

	def getNumericEdges(self):
		tmpdb = []
		print("Converting into numerical edges...")
		for i in self.edges:
			tmpdb.append((self.nodes1.index(i[0]), self.nodes2.index(i[1])))
		return tmpdb
	
	def addMissingConnections(self):
		present = set()
		ne = len(self.edges)
		for e in self.edges:
			present.add(e[0])
		missing = set(self.nodes1).difference(present)
		mismatched = set(self.nodes1).symmetric_difference(present).difference(missing)
		print("Found %d nodes1 with no connections, %d mismatched edges" % (len(missing), len(mismatched)))
		self.removeEdges(list(mismatched))
		if len(missing) > 0:
			self.nodes2.append("dummyNode")
			for e in missing:
				self.edges.append((e, "dummyNode"))	
			print("Added %d edges" % (len(self.edges) - ne))
			return ["dummyNode"]
		else:
			return []

	def pruneNodes2(self):
		present = set()
		for e in self.edges:
			present.add(e[1])
		missing = list(set(self.nodes2).difference(present))
		assert set(missing) == set(self.nodes2).symmetric_difference(present)
		print("Found %d nodes2 with no connections, removing them..." % len(missing))
		print(missing)
		self.removeNodes([], missing)
		return missing

	def removeEdges(self, edges1, edges2=[]):
		tmp = []	
		for n in edges1:
			for e in self.edges:
				if e[0] == n: #TODO PROB I SHOULDNT ITERATE WHILE REMOVING
					tmp.append(e)
		for e in tmp:
			self.edges.remove(e)
		tmp = []
		for n in edges2:
			for e in self.edges:
				if e[1] == n:
					tmp.append(e)
		for e in tmp:
			self.edges.remove(e)

	def removeNodes(self, nodes1, nodes2=[]):
		self.nodes1.remove(nodes1)
		self.removeEdges(nodes1)
		self.nodes2.remove(nodes2)
		self.removeEdges([], nodes2)

	def addNodes(self, nodes1, nodes2=[]):
		self.nodes1 = self.nodes1 + nodes1
		self.nodes2 = self.nodes2 + nodes2

	def checkSanity(self):
		missing1 = []
		for n in self.nodes1:
			found = False
			for e in self.edges:
				if n in e[0]:
					found = True
					break
			if not found:
				missing1.append(n)
		return missing1

	def removeNodesNotIn(self, nodes1, nodes2=[]):
		c = 0
		ec = 0
		for n in self.nodes1:
			if n not in nodes1:
				self.nodes1.remove(n)
				c+=1
				for e in self.edges:
					if e[0] == n:
						self.edges.remove(e)
						ec +1
		print("Removed %d nodes1 and %d edges"%(c, ec))
		c = 0
		ec = 0
		for n in self.nodes2:
			if not n in nodes2:
				self.nodes2.remove(n)
				c+=1
				for e in self.edges:
					if e[1] == n:
						self.edges.remove(e)
						ec+=1
		print("Removed %d nodes2 and %d edges"%(c, ec))

	def size(self):
		return t.Size([len(self.nodes1), len(self.nodes2)])

	def getDataForCOO(self):
		numericEdges = self.getNumericEdges()
		return t.tensor(numericEdges).t(), t.Size([len(self.nodes1), len(self.nodes2)])

	def getAdjMatrix(self, values="ones"):
		v = None
		if values == "ones":
			v = t.ones(len(self.edges))
		elif values == "random":
			v = t.normal(0, 0.01, [len(self.edges)])
		else:
			v = t.empty(len(self.edges))
		#print(self.edges)
		adj = t.sparse_coo_tensor(t.tensor(self.edges).t(), v, t.Size([len(self.nodes1), len(self.nodes2)]))
		return adj

def readEdgesFromFile(f):
	ifp = open(f, "r")
	lines = ifp.readlines()
	ifp.close()
	edges = set()
	for l in lines:
		tmp = l.strip().split()
		edges.add((tmp[0], tmp[1]))
	return list(edges)

class BiologicalHierarchy(object):
"""
Class, taking in information on (possibly multiple) biological networks (in the form of a lists of edges or a BiologicalNetwork objects) 
and making them suitable to build biologically meaningful sparsified neural network layers. 
For subsequent layers, it will make sure the imput nodes of the layer have at least 1 connection, otherwise connecting them to the dummy node, 
and that all output nodes have at least one input node, otherwise removing the output node.

"""
	def __init__(self, edgesLists, harmonizeNets=True):
		self.bioNets = []
		#print(edgesLists)

		for i, e in enumerate(edgesLists):
			if i == 0:
				assert type(e) == BiologicalNetwork #the order of nodes1 must be fixed at least for the first layer
			if type(e) == str:
				print("Reading %s" % e)
				tmp = readEdgesFromFile(e)
				self.bioNets.append(BiologicalNetwork(tmp, self.bioNets[i-1].nodes2))
			elif type(e) == list:
				print("Reading list of edges")
				self.bioNets.append(BiologicalNetwork(e, self.bioNets[i-1].nodes2))
			elif type(e) == BiologicalNetwork:
				print("Reading BioNet obj")
				self.bioNets.append(e)
		print("Found %d nets" % len(self))
		#tmpnodes1 = self.bioNets[0].nodes1
		a = 0
		for n in self.bioNets:
			#print(a,"----------")
			#n.nodes1 = tmpnodes1
			#print(self)
			#check that all the nodes1 have 1 connection
			#add nodes1 connections to dummy nodes2 if necessary
			n.addMissingConnections()
			#check that all the nodes2 have 1 connection
			#remove nodes2 with no connection if necessary
			n.pruneNodes2()
			#pass updated nodes2 to next network
			#tmpnodes1 = n.nodes2
			a+=1
		print("Final harmonized network:")
		print(self)

	def __getitem__(self, i):
		return self.bioNets[i]

	def __str__(self):
		r = "%d nets\n" % len(self.bioNets)
		for i in self.bioNets:
			r += str(i)+"\n"
		return r

	def __len__(self):
		return len(self.bioNets)

	def dump(self, name):
		pickle.dump(self, open(name, "wb"))
