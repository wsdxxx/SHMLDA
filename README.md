# SHMLDA

1. Directory Structure

main.py 
#Entry for main training and evaluation

model.py                             
#Encoder, decoder, and GMAE training framework

SoftHGNN.py                          
#Core implementation of soft hypergraph and dual-channel structural encoder

mask.py                              
#Edge masking and sampling

utils.py                             
#Data loading, similarity graph construction, negative sampling

2. Datasets

Four benchmark datasets

`LCS.csv`  lncRNA cosine similarity

`LGS.csv`  lncRNA GIPK similarity

`LFS.csv`  lncRNA functional similarity

`DCS.csv`  disease cosine similarity

`DGS.csv`  disease GIPK similarity

`DSS.csv`  disease semantic similarity

`LD_adjmat.csv`  lncRNA–disease association adjacency matrix (0/1)

4. Environment

Python  3.9

torch   2.3.0

numpy   1.26.4

pandas  2.2.2

sklearn 1.5.1

scipy   1.13.1

matplotlib 3.9.2

5. Quick Start

Main experiment:

python main.py --dataset 1 --evaluation_protocol 5cv

Independent test protocol:

python main.py --dataset 1 --evaluation_protocol independent
