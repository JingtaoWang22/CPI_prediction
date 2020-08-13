#!/bin/bash

DATASET=human
#DATASET=celegans
# DATASET=yourdata
dataname=H_1:1

# radius=0  # w/o fingerprints (i.e., atoms).
# radius=1
radius=2
# radius=3

# ngram=2
ngram=1

python preprocess_data.py $DATASET $dataname  $radius $ngram
python preprocess_data.py $DATASET H_1:3  $radius $ngram
python preprocess_data.py $DATASET H_1:5  $radius $ngram
python preprocess_data.py celegans C_1:1  $radius $ngram
python preprocess_data.py celegans C_1:3  $radius $ngram
python preprocess_data.py celegans C_1:5  $radius $ngram
