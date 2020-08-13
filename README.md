# Compound-protein interaction (CPI) prediction using GNN and Protein Transformer

The code is based on Dr.Tsubaki's repository here: https://github.com/masashitsubaki/CPI_prediction, which has the implementation of the paper 
"[Compound-protein Interaction Prediction with End-to-end Learning of Neural Networks for Graphs and Sequences (Bioinformatics, 2018)](https://academic.oup.com/bioinformatics/advance-article-abstract/doi/10.1093/bioinformatics/bty535/5050020?redirectedFrom=PDF)". I am looking for a better model for protein and transformer seems to be a strong candidate.


In this repository, these CPI datasets are provided: 

1. The datasets in Dr.Tsubaki's repository, which are human and *C. elegans* created by Liu et al. in this paper:
"[Improving compound–protein interaction prediction by building up highly credible negative samples (Bioinformatics, 2015).](https://academic.oup.com/bioinformatics/article/31/12/i221/216307)"
Note that the ratio of positive and negative samples is 1:1.

2. Datasets from the same paper of ratio 1:1, 1:3, and 1:5 made by myself. Note that the 1:1 datasets are almost identical to Dr.Tsubaki's except a few (about 10) compound protein pairs are missing. This is due to the updating of Uniprot accession used in Liu et al.'s datasets. See here for detail: https://www.uniprot.org/help/deleted_accessions.

In the problem setting of CPI prediction,
an input is the pair of a SMILES format of compound and an amino acid sequence of protein;
an output is a binary label (interact or not).
The SMILES is converted with RDKit and
we obtain a 2D graph-structured data of the compound (i.e., atom types and their adjacency matrix).


## Requirements

- PyTorch
- scikit-learn
- RDKit
- numpy





