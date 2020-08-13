#!/bin/bash

DATASET=human
#DATASET=celegans
# DATASET=yourdata

dataname=H_1:3

radius=2

ngram=3

dim=10

heads=2

dim_ff=10

N=1   #number of self-attn layers

M=1   #number of tgt_attn layers

layer_gnn=3

window=5  # The window size is 2*window+1.
layer_output=3
layer_cnn=3

lr=1e-3

lr_decay=0.7

decay_interval=10

iteration=100

warmup_step=50

dropout=0
weight_decay=1e-6


setting=M3_N1
python m3.py $DATASET $dataname $radius $ngram $dim $dim_ff $layer_gnn $layer_output $heads $N $M $lr $lr_decay $decay_interval $weight_decay   $iteration $warmup_step  $dropout $setting

N=2

setting=M3_N2
python m3.py $DATASET $dataname $radius $ngram $dim $dim_ff $layer_gnn $layer_output $heads $N $M $lr $lr_decay $decay_interval $weight_decay   $iteration $warmup_step  $dropout $setting


setting=M2_N2_M1
python m2.py $DATASET $dataname $radius $ngram $dim $dim_ff $layer_gnn $layer_output $heads $N $M $lr $lr_decay $decay_interval $weight_decay   $iteration $warmup_step  $dropout $setting

setting=M1_N2_M1
python m1.py $DATASET $dataname $radius $ngram $dim $dim_ff $layer_gnn $layer_output $heads $N $M $lr $lr_decay $decay_interval $weight_decay   $iteration $warmup_step  $dropout $setting

N=1

setting=M2_N1_M1
python m2.py $DATASET $dataname $radius $ngram $dim $dim_ff $layer_gnn $layer_output $heads $N $M $lr $lr_decay $decay_interval $weight_decay   $iteration $warmup_step  $dropout $setting

setting=M1_N1_M1
python m1.py $DATASET $dataname $radius $ngram $dim $dim_ff $layer_gnn $layer_output $heads $N $M $lr $lr_decay $decay_interval $weight_decay   $iteration $warmup_step  $dropout $setting

N=2
M=2

setting=M2_N2_M2
python m2.py $DATASET $dataname $radius $ngram $dim $dim_ff $layer_gnn $layer_output $heads $N $M $lr $lr_decay $decay_interval $weight_decay   $iteration $warmup_step  $dropout $setting

setting=M1_N2_M2
python m1.py $DATASET $dataname $radius $ngram $dim $dim_ff $layer_gnn $layer_output $heads $N $M $lr $lr_decay $decay_interval $weight_decay   $iteration $warmup_step  $dropout $setting

