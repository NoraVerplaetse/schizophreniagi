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

import sys, os, pickle
import numpy as np
import math
from hilbertcurve.hilbertcurve import HilbertCurve
from scipy.sparse import coo_matrix

count_nonsyn = True
features_withscores = {"UTR3":0, "UTR5":1, "splicing":2, "upstream":3, "downstream":4, "intronic":5, "ncRNA_exonic":6, "ncRNA_intronic":7, "ncRNA_splicing":8, "exonic_nonframeshift_insertion":9, "exonic_nonframeshift_deletion":10, "exonic_stoploss":11,  "exonic_stopgain":12, "exonic_frameshift_insertion":13, "exonic_frameshift_deletion":14,  "exonic_startloss":15, "exonic_nonsynonymousSNV": 16,"provean_low":17, "provean_mid":18, "provean_high":19, "mcap_low":20, "mcap_mid":21, "mcap_high":22, "metasvm_low":23, "metasvm_mid":24, "metasvm_high":25, "deogen_low":26, "deogen_mid":27, "deogen_high":28, "RVIS":29, "GDI":30, "pHI":31, "pRec":32}
features_noscores = {"UTR3":0, "UTR5":1, "splicing":2, "upstream":3, "downstream":4, "intronic":5, "ncRNA_exonic":6, "ncRNA_intronic":7, "ncRNA_splicing":8, "exonic_nonframeshift_insertion":9, "exonic_nonframeshift_deletion":10, "exonic_stoploss":11,  "exonic_stopgain":12, "exonic_frameshift_insertion":13, "exonic_frameshift_deletion":14, "exonic_startloss":15, "exonic_nonsynonymousSNV": 16}

def parseMultiVCF(multivcffile, nbvariants, ids, genomeversion='hg19', SNPvector=True, remove_samevaluesnp =True, zygosityHC=True):
        """
        This function reads a multiVCF file containing the zygosities for all snps for all samples.

        If SNPvector == True, it will store the zygosities for all variants in the dataset in a feature vector and this for all samples present in ids. 
        Value zero means variant is not present or missing, value 1 means variant is heterozygous, value 2 homozygous.
        If remove_samevaluesnp=True, the variants with the same value for all the samples will be removed.
        If zygosityHC == True, for each sample id, it will create a hilbert curve in the form of sparse coo_matrix with the genomic coordinates as 2D coordinates in the image, and the zygosity as value. 
        Value zero means variant is not present or missing, value 1 means variant is heterozygous, value 2 homozygous.
        """
        snpdatamx=np.zeros((len(ids), nbvariants), dtype=np.int8)
        variants=[]
        count_heterozyg=0
        count_homozyg=0

        distances={sampleid:[] for sampleid in ids}
        zygosities={sampleid:[] for sampleid in ids}
        if genomeversion=='hg19':
                chromlengths=[249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566,155270560,59373566, 16569] #chrom 1-22, X, Y, MIT
        cumsum_chrom=list(np.cumsum(chromlengths))
        cumsum_chrom.insert(0,0)
        bits=math.ceil(math.log(math.sqrt(cumsum_chrom[-1]),2))
        HC = HilbertCurve(bits,2,n_procs=0)

        with open(multivcffile) as vcffile:
                linenr=0
                line=vcffile.readline()
                while len(line)>0:
                        if "#" in line:
                                line=vcffile.readline()
                                continue
                        tmp=line.split("\t")
                        variants.append((str(tmp[0]) + "_" + str(tmp[1]), tmp[2]))

                        chrom=tmp[0]
                        if chrom =='X' or chrom =='Y' or chrom == 'MT':
                                linenr+=1
                                line=vcffile.readline()
                                continue
                        chrom = int(chrom)
                        pos=int(tmp[1])
                        varid=tmp[2]
                        ref=tmp[3]
                        alt=tmp[4]

                        for idx,i in enumerate(range(9, len(ids)+9)):
                                if tmp[i].split(":")[0] != "0/0" and tmp[i].split(":")[0] != "./0" and tmp[i].split(":")[0] != "0/." and tmp[i].split(":")[0] != "./.":
                                        if "0" in tmp[i].split(":")[0] or "." in tmp[i].split(":")[0]:
                                                if SNPvector:
                                                        snpdatamx[idx][linenr]=1
                                                        count_heterozyg+=1

                                                        if zygosityHC:
                                                                distances[ids[idx]].append(int(cumsum_chrom[chrom-1])+pos)
                                                                zygosities[ids[idx]].append(1)
                                        else:
                                                if SNPvector:
                                                        snpdatamx[idx][linenr]=2
                                                        count_homozyg+=1
                                                if zygosityHC:
                                                        distances[ids[idx]].append(int(cumsum_chrom[chrom-1])+pos)
                                                        zygosities[ids[idx]].append(2)
                        line=vcffile.readline()
                        linenr+=1
                print('Number of heterozygous SNPs: ', count_heterozyg, " NUmber of homozygous SNPs: ", count_homozyg)

        if SNPvector == True:
                if remove_samevaluesnp == True:
                        samevaluecolumns=[]
                        for j in range(0, nbvariants):
                                if np.all(snpdatamx[:,j] == snpdatamx[:,j][0]):
                                        samevaluecolumns.append(j)
                        newdatamx = np.delete(snpdatamx, samevaluecolumns, 1)
                        keptvariants=[x for x in variants if variants.index(x) not in samevaluecolumns]
                else:
                        newdatamx = snpdatamx
                        keptvariants = variants

                np.save("SNPvector.npy", newdatamx)
                print('datamatrices without and with removing same value SNPs', snpdatamx.shape, newdatamx.shape)

        if zygosityHC == True:
                db=[]
                for idx,j in enumerate(ids):#range(len(ids)):
                        points=HC.points_from_distances(distances[j])
                        points= np.array(points, dtype=np.uint32)
                        db.append(coo_matrix((zygosities[j], (points[:,0], points[:,1])), shape=(2**(bits), 2**bits)))
                        print("processed sample ", idx)
                np.save("HCzyg.npy", db)
                print(len(db))

        if SNPvector and zygosityHC:
                return newdatamx, keptvariants, zygosities, distances, db
        elif SNPvector:
                return newdatamx, keptvariants
        elif zygosityHC:
                return zygosities, distances, db	   

def parserGC(file, genelist, features=features_noscores, snpidsmissing={}, missenseOnly=False, onlyRegions=["exonic"]):
        """This function reads VCF file, annotated by annovar, and creates a matrix (numpy ndarray) containing information on variants per gene."""
        ifp = open(file)
        line = ifp.readline()
        assert "Chr" in line and "Start" in line
        line = ifp.readline()

        missinggenes=[]
        linesmoregenesthanregions=[]
        exonicwithoutregion=0

        db=np.zeros((len(genelist), len(features)), dtype=np.float32)
        linenr=1

        while len(line)>0:
                tmp = line.split("\t")
                crom = tmp[0]
                pos = int(tmp[1])
                mut = (tmp[3],tmp[4])

                region = tmp[5].split(";")
                gene= tmp[6].split(";")

                for i in region:
                        if (onlyRegions != None and len(onlyRegions) >0) and not i in onlyRegions:
                                linenr+=1
                                continue

                        if "intergenic" in i:
                                linenr+=1
                                continue

                        vartype = tmp[8]

                        if i == "exonic" and vartype == "synonymous SNV":
                                linenr+=1
                                continue

                        if i == "exonic" and len(vartype)==0:
                                linenr+=1
                                continue

                        if i == "exonic" and "unknown" in vartype:
                                linenr+=1
                                continue

                        if missenseOnly and i == "exonic" and not "nonsynonymous" in vartype:
                                linenr+=1
                                continue

                        vartype = vartype.replace(" ", "_")

                        #in case more than region we pair the region with gene at same position
                        if len(region)>1:
                                if len(gene) != len(region):
                                        linesmoregenesthanregions.append(line)
                                        if region.index(i)==0:
                                                genematch = gene[0]
                                        else:
                                                genematch=gene[-1]
                                else:
                                        genematch = gene[region.index(i)]
                                        
                                if not genematch in genelist: #genelist.has_key(gene):
                                        print("Error: gene %s not in genelist" %genematch)
                                        missinggenes.append(genematch)
                                        linenr+=1
                                        continue
                                        
                                if i == "exonic":
                                        if vartype == "nonsynonymous_SNV":
                                                if count_nonsyn == True:
                                                        db[genelist[genematch]][features["exonic_nonsynonymousSNV"]] +=1     
                                        else:
                                                if len(vartype) == 0:
                                                        exonicwithoutregion+=1
                                                        print('exonic without region',line)
                                                else:
                                                        db[genelist[genematch]][features[i+"_"+vartype]] += 1
                                else:
                                        db[genelist[genematch]][features[i]] += 1
                        #in case only 1 region, possibly with more than one gene
                        else:
                                for g in gene:

                                        if not g in genelist: #genelist.has_key(gene):
                                                print("Error: gene %s not in genelist" %g)
                                                missinggenes.append(g)
                                                continue

                                        if i == "exonic":
                                                if vartype == "nonsynonymous_SNV":
                                                        if count_nonsyn == True:
                                                                db[genelist[g]][features["exonic_nonsynonymousSNV"]] +=1
                                                else:

                                                        if len(vartype) == 0:
                                                                exonicwithoutregion+=1
                                                                print('exonic without region', line)
                                                        else:
                                                                db[genelist[g]][features[i+"_"+vartype]] += 1
                                        else:
                                                if i in features:
                                                        db[genelist[g]][features[i]] += 1
                line = ifp.readline()
                linenr+=1
                
        return db

def parserMutlist(file, genelist, snpidsmissing={}, missenseOnly=True, onlyRegions=["exonic"], variantTypesHT=features_noscores):
        """This function reads VCF file, annotated by annovar, and creates a matrix (numpy ndarray) containing information on variants per gene."""
        ifp = open(file)
        line = ifp.readline()
        assert "Chr" in line and "Start" in line
        line = ifp.readline()
        numMuts = 0

        values=np.zeros((15000,10), dtype=np.float32)
        linenr=1

        while len(line)>0:
                tmp = line.split("\t")
                crom = tmp[0]
                pos = int(tmp[1])
                mut = (tmp[3],tmp[4])

                if crom == 'MT':
                        linenr+=1
                        line=ifp.readline()
                        continue
                if crom == 'X':
                        crom = 23
                if crom == 'Y':
                        linenr+=1
                        line=ifp.readline()
                        continue
                crom = int(crom)
                region = tmp[5].split(";")[0]
                gene= tmp[6].split(";")[0]

                if (onlyRegions != None and len(onlyRegions) >0) and not region in onlyRegions:
                        linenr+=1
                        line=ifp.readline()
                        continue
                vartype = tmp[8].split(";")[0]

                #not interested in unknown or synonymous exonic variants
                if "unknown" in vartype or vartype == "":
                        line=ifp.readline()
                        linenr+=1
                        continue
                if " SNV" in vartype:
                        vartype = vartype.split(" ")[0]
                else:
                        vartype = vartype.replace(" ", "_")

                if  missenseOnly and "exonic" in region and ("synonymous" == vartype or "." == vartype):
                        line=ifp.readline()
                        linenr+=1
                        continue
                if not gene in genelist:
                        line=ifp.readline()
                        continue
                RVIS =  parseScore(tmp[22])
                if RVIS != -1:
                        RVIS= RVIS/100
                GDI = parseScore(tmp[24])
                pHI = parseScore(tmp[19])
                pRec = parseScore(tmp[20])
                vest4 = parseScore(tmp[53])
                metasvm = parseScore(tmp[55])
                deogen = parseScore(tmp[78])
                phylop30 = parseScore(tmp[122])
                varEff = [variantTypesHT[region+"_"+vartype], pHI, pRec, RVIS, GDI, metasvm, vest4, deogen, phylop30]

                genePos = genelist[gene]
                if numMuts<10:
                        print(gene, genelist[gene])
                if numMuts >= 15000:
                        print("Nummuts larger than 15000")
                        print("*************************************************************")
                        break

                if region == "exonic":
                        values[numMuts] = [float(genePos)]+varEff

                        if numMuts <9:
                                print(genePos, varEff)
                                print([crom, genePos]+varEff)
                                print('values', values)
                elif region in variantTypesHT:
                        raise Exception("Not implemented for non coding variants")
                else:
                        line=ifp.readline()
                        continue
                        
                numMuts +=1
                line=ifp.readline()
        print(values.shape)
        print("Found %d mutations:" % (numMuts))
        return values, numMuts

def castFloat(v):
        """ Function to process score cell value, casting multiple scores into one value (the average) or returning -1000 for missing values """
        a = None
        #take average if multiple scores for multiple transcripts
        if ";" in str(v):
                scores=v.split(";")
                i=0
                sum=0
                for s in scores:
                        if s != ".":
                                sum += float(s)
                                i+=1
                if i>0:
                        a=sum/i
                else:
                        return -1000
        else:
                try:
                        a = float(v)
                except:
                        return -1000
        return a

def parseScore(s):
        """ Function to process score cell value, casting multiple scores into one value (first value) or returning -1 for missing values """
        tmp=s.strip().split(";")
        if tmp[0] == '' or tmp[0] ==".":
                score=-1
        else:
                score = float(tmp[0])
        return score
        
def mainSNPvectorAndHC():
        """ Main function to create snp vector containing 0,1,2 values for all variants depending on the presence, heterozygosity of homozygosity of the variant. 
                            hilbert curves with genomic coordinates as 2D pixel coordinate and zygosity as value (0,1,2 values for all variants depending on the presence, heterozygosity of homozygosity of the variant)"""
        ids = pickle.load(open("ids_multivcf.pickle", 'rb'))
        multivcffile =  "multivcf.vcf"
        nbvariants = 1811204
        snpvector, keptvariants, zygosities, distances, HCzygosity = parseMultiVCFintoSNPvector(multivcffile, nbvariants, ids, genomeversion='hg19', remove_samevaluesnp =True, zygosityHC=True)
        
        return snpvector, HCzygosity

def mainGC():
        """ Main function to create gene-centric feature matrix"""
        removezerogenes=False
        genesdict=pickle.load(open("genesdictRefGeneAnnovarComplete_withoutzerogenes", 'rb'))
        genes = {genesdict[i]:i for i in genesdict.keys()}
        listgenomes =[]
        sampleids= pickle.load(open("ids_multivcf.pickle", 'rb'))

        #create ndarray for each sample id
        snpidsmissing={}
        for index,i in enumerate(sampleids):
                print(index)
                file = "annotatedfile_" + i + ".avinput.hg19_multianno.txt"
                db, snpidsmissing = parserGC(file, genes, features=features_noscorces, variantscores = False, genescores = False, missenseOnly=False, onlyRegions=None, snpidsmissing=snpidsmissing)
                listgenomes.append(db)

        if removezerogenes == True:
                genes=list(genesdict.values())
                print(len(genes), genes[0])
                listgenomes = np.take(listgenomes, list(range(17)))
                geneszerorows=[] 
                for j in range(0,listgenomes.shape[1]):
                        sumrows=sum(listgenomes[0][j])
                        if sumrows == 0:
                                geneszerorows.append(j)

                #check for each sample and only keep genes that consistently sum to zero
                for i in range(1,listgenomes.shape[0]):
                        genessumzero=[]
                        for j in range(0,listgenomes.shape[1]):
                                sumrows=sum(listgenomes[i][j])
                                if sumrows == 0:
                                        genessumzero.append(j)
                        geneszerorows=[value for value in geneszerorows if value in genessumzero]
                        print(i)

                print(len(geneszerorows))
                print('genes',len(genes))

                genestodelete=[genes[index] for index in geneszerorows]
                for i in genestodelete:
                        genes.remove(i)

                print('genes after deleting zero rows',len(genes))

                newdict={key:value for key,value in enumerate(genes)}                   
                newfeatvect = np.delete(listgenomes, geneszerorows, 1)
                return newfeatvect          

        return np.array(listgenomes)

def mainMutlist():
        """ Main function to create mutation list feature matrix together with lengths array that represents the unique number of variants in each sample."""
        TEST = False
        genesdict=pickle.load(open("genesdictRefGeneAnnovarComplete_withoutzerogenes", 'rb'))
        genes = {genesdict[i]:i for i in genesdict.keys()}
        listgenomes =[]
        sampleids= pickle.load(open("ids_multivcf.pickle", 'rb'))
        db=np.zeros((len(sampleids), 15000,10), dtype=np.float32)
        lens=[]

        #create ndarray for each sample id
        snpidsmissing={}
        for index,i in enumerate(sampleids):
                print(index, i)
                annfile = "annotatedfile_" + i + ".avinput.hg19_multianno.txt"
                db[index],l = parserMutlist(file, genes, missenseOnly=True, onlyRegions=["exonic"])
                lens.append(l)

        return np.array(db, dtype=np.float32), np.array(lens)