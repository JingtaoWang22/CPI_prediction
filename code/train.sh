#!/bin/bash

DATASET=celegans
#DATASET=celegans
# DATASET=yourdata

dataname=celegans1:1


radius=2
ngram=3
dim=10
heads=2
dim_ff=10
N=3   #number of self-attn layers
M=1   #number of tgt_attn layers
layer_gnn=3
window=5  # The window size is 2*window+1.
layer_output=1
layer_cnn=3
lr=2e-4
lr_decay=0.5
decay_interval=10
iteration=100
warmup_step=50
dropout=0
weight_decay=0
#1e-6




N=3
M=1
setting=$DATASET--radius$radius--ngram$ngram--dim$dim--layer_gnn$layer_gnn--window$window--layer_cnn$layer_cnn--layer_output$layer_output--N_encoder$N--M_decoder$M--lr$lr--lr_decay$lr_decay--decay_interval$decay_interval--weight_decay$weight_decay--iteration$iteration

python train_model.py $DATASET $dataname $radius $ngram $dim $dim_ff $layer_gnn $layer_output $heads $N $M $lr $lr_decay $decay_interval $weight_decay   $iteration $warmup_step  $dropout $setting

