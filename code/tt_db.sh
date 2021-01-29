#!/bin/bash

DATASET=BindingDB
#DATASET=celegans
# DATASET=yourdata

dataname=null

#DATA
radius=2
ngram=3

#CNN,GNN
layer_gnn=3
window=5  # The window size is 2*window+1.
layer_output=1
layer_cnn=3

#TRANSFORMER
dim=$1
heads=$2
dim_ff=$3
N=$4   #number of self-attn layers
M=$5   #number of tgt_attn layers
warmup_step=30
dropout=0

#MODEL:
lr=15e-5
lr_decay=0.5
decay_interval=10
iteration=100
weight_decay=0
#1e-6



python training_db.py $DATASET $dataname $radius $ngram $dim $dim_ff $layer_gnn $layer_output $heads $N $M $lr $lr_decay $decay_interval $weight_decay  $iteration $warmup_step  $dropout $setting




