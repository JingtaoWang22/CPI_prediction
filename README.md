# Compound-protein interaction (CPI) prediction using GNN and Protein Transformer

The code is based on Dr.Tsubaki's repository here: https://github.com/masashitsubaki/CPI_prediction, which has the implementation of the paper 
"[Compound-protein Interaction Prediction with End-to-end Learning of Neural Networks for Graphs and Sequences (Bioinformatics, 2018)](https://academic.oup.com/bioinformatics/advance-article-abstract/doi/10.1093/bioinformatics/bty535/5050020?redirectedFrom=PDF)". It uses Graph Neural Network for learning the representation of protein. In this repository, I implement Protein Transformer for learning protein representation. It takes the advantage of self-attention and capture the long-distance interaction between amino acid residues. This brings about significant improvements in predictions power.

In this repository, the CPI datasets of human and *C. elegans* are created by Liu et al. in this paper:
"[Improving compound–protein interaction prediction by building up highly credible negative samples (Bioinformatics, 2015).](https://academic.oup.com/bioinformatics/article/31/12/i221/216307)"

In the problem setting of CPI prediction,
an input is the pair of a SMILES format of compound and an amino acid sequence of protein;
an output is a binary label (interact or not).
The SMILES is converted with RDKit and
we obtain a 2D graph-structured data of the compound (i.e., atom types and their adjacency matrix).

Representation of proteins and compounds learned by 2 sub-networks are concatenated to predict the interaction.

## Requirements

- PyTorch
- scikit-learn
- RDKit
- numpy=1.16.1





