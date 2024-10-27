#!/bin/bash

model_list="swinunet swinnet"
dataset_list="tn3k" 

fold_list="0 1 2 3 4"

for model in $model_list
do
    for fold in $fold_list
    do
        CUDA_VISIBLE_DEVICES=1 python train.py -loss_func ce -backbone "$model" -fold "$fold" -dataset "$dataset_list"
    done
    CUDA_VISIBLE_DEVICES=1 python inference-test.py -backbone "$model" -dataset "$dataset_list"
done
