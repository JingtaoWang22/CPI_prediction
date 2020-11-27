#!/bin/bash

DATASET=celegans
#DATASET=human
# DATASET=yourdata

dataname=celegans1:1

radius=2

ngram=3

python preprocess_data.py $DATASET $dataname  $radius $ngram

