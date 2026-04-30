# What is this repository for? #
SchizophreniaGI is a Neural Network based Genome Interpretation (GI) framework for the exome-based in-silico discrimination of healthy controls and schizophrenia patients. More details will be soon available in: Verplaetse, N., Moreau, Y., Raimondi, D. XXX. The code here contains a standalone version of SchizophreniaGI, which takes as input an ANNOVAR-annotated .txt version of a VCF file and outputs the predicted likelihood of schizophrenia.

# Where can I find the data? #
Access to the data used can be requested through dbGaP (Sweden-Schizophrenia Population-Based Case-Control Exome Sequencing project dbGaP phs000473.v2.p).

# What does this repository contain? #
main_MLbaseline.py -> main predictive function for the linear baseline

main_NN.py -> main predictive function for the different neural net architectures: NNlogreg, NNlinear, NNsmalldense NNlargedense, NNbiosparseKEGG, NNbiosparseCPATHDB, NNdo, GCNmutlist, TNNmutlist, CNNhc Args: gpu_devicename (str) weight_decay (float) dropout (float) penalty (str): 'l1','l2' batch_size (int) epochs (int)

parsers.py -> functions to parse annotated VCF files from Annovar into the different representations parserMultiVCF: naieve SNPvector and Hilbert curve zygosity image parser parserGC: gene-centric encoding parser parserMutlist: mutation list encoding parser

sources/models.py -> neural net layers and architectures for gene-centric models, mutation list models and Hilbert curve models.

sources/BioNets.py -> code for building biological sparse networks to build neural network layers from

sources/wrappers.py -> neural net wrappers with dataset, fit and predict functions for gene-centric models, mutation list models and Hilbert curve models

sources/utils.py -> scaler function for gene-centric representation on GPU

toyExample/ -> containing test data

requirements.yml 

README.md -> this readme

# How do I annotate exomes with ANNOVAR? #
These are the links to Annovar documentaiton (http://annovar.openbioinformatics.org/en/latest/) and installation instructions (http://annovar.openbioinformatics.org/en/latest/user-guide/download/). To annotate the multiVCF from dbGaP, we ran the following command line :

perl .../annovar/table annovar.pl VCFFILE .../annovar/humandb/ -out OUTPUT -vcfinput -buildver hg19 -protocol refGene,dbnsfp42a -operation gx,f -xreffile XREFFILE where OUTPUT is the Annovar file containing the annotations. The database used for the annotation is dbNSFP version 42a (https://sites.google.com/site/jpopgen/dbNSFP).

# Who do I talk to? #
daniele DoT raimondi At igmm DoT cnrs DoT fr nora DoT verplaetse aT kuleuven DoTbe yves dOt moreau At kuleuven dOt be