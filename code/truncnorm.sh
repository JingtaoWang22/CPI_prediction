#!/bin/bash

DATASET=human
# DATASET=celegans
# DATASET=yourdata

dataname=H_1:3

radius=2

ngram=1

dim=$1

heads=$2

dim_ff=$3

dim_ff2=$4

N=$5   #number of self-attn layers

M=$6   #number of tgt_attn layers

layer_gnn=3

window=5  # The window size is 2*window+1.

layer_cnn=3

lr=1e-4

lr_decay=0.7

decay_interval=10

iteration=100

setting=$DATASET--radius$radius--ngram$ngram--dim$dim--heads$heads--dim_ff$dim_ff--dim_ff2$dim_ff2--N$N--M$M--layer_gnn$layer_gnn--window$window--layer_cnn$layer_cnn--lr$lr--lr_decay$lr_decay--decay_interval$decay_interval

python truncnorm.py $DATASET $dataname $radius $ngram $dim $heads $dim_ff $dim_ff2 $N $M $layer_gnn $window $layer_cnn $lr $lr_decay $decay_interval $iteration $setting

