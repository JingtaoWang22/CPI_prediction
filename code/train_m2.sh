#!/bin/bash

DATASET=human
#DATASET=celegans
# DATASET=yourdata

dataname=H_1_3
radius=2
ngram=3
dim=10
heads=2
dim_ff=10
N=$1   #number of self-attn layers
M=$2   #number of tgt_attn layers
layer_gnn=3
window=5  # The window size is 2*window+1.
layer_output=1
layer_cnn=3
lr=4e-4
lr_decay=0.5
decay_interval=10
iteration=100
warmup_step=50
dropout=0
weight_decay=0
#1e-6


setting=M2_N$1_M$2
python self_cnn_decoder.py $DATASET $dataname $radius $ngram $dim $dim_ff $layer_gnn $layer_output $heads $N $M $lr $lr_decay $decay_interval $weight_decay   $iteration $warmup_step  $dropout $setting


