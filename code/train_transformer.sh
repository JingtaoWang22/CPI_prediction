#!/bin/bash

DATASET=human
#DATASET=celegans
# DATASET=yourdata

dataname=H_1:3

#DATA
radius=2
ngram=3

#CNN,GNN
layer_gnn=2
window=5  # The window size is 2*window+1.
layer_output=1
layer_cnn=2

#TRANSFORMER
dim=$1
heads=$2
dim_ff=$3
N=$4   #number of self-attn layers
M=$5   #number of tgt_attn layers
warmup_step=50
dropout=0

#MODEL:
lr=1e-3
lr_decay=0.7
decay_interval=10
iteration=100
weight_decay=0
#1e-6



python training.py $DATASET $dataname $radius $ngram $dim $dim_ff $layer_gnn $layer_output $heads $N $M $lr $lr_decay $decay_interval $weight_decay  $iteration $warmup_step  $dropout $setting




