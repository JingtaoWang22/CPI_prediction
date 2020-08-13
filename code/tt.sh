#!/bin/bash

DATASET=human
#DATASET=celegans
# DATASET=yourdata

dataname=H_1:3

radius=2

ngram=3

dim=$1

heads=$2

dim_ff=$3

N=$4   #number of self-attn layers

M=$5   #number of tgt_attn layers

layer_gnn=3

window=5  # The window size is 2*window+1.
layer_output=3
layer_cnn=3

lr=1e-3

lr_decay=0.7

decay_interval=10

iteration=100

warmup_step=50

dropout=0.2
weight_decay=1e-6


setting=$DATASET--radius$radius--ngram$ngram--dim$dim--heads$heads--dim_ff$dim_ff--dim_ff2$dim_ff2--n_encoder$N--n_decoder$M--layer_gnn$layer_gnn--window$window--layer_cnn$layer_cnn--lr$lr--lr_decay$lr_decay--decay_interval$decay_interval--dropout$dropout

python training.py $DATASET $dataname $radius $ngram $dim $dim_ff $layer_gnn $layer_output $heads $N $M $lr $lr_decay $decay_interval $weight_decay   $iteration $warmup_step  $dropout $setting




