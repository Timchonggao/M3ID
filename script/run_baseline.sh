#!/bin/bash

model_list="resnet densenet convnext vit logonet"  # 可以扩展为 "resnet densenet convnext vit"
dataset_list="tn3k"  # 可以扩展为 "tn3k tndt"
fold_list="0 1 2 3 4"

for model in $model_list
do
    for fold in $fold_list
    do
        CUDA_VISIBLE_DEVICES=0 python train.py -loss_func ce -backbone "$model" -fold "$fold" -dataset "$dataset_list"
    done
    CUDA_VISIBLE_DEVICES=0 python inference-test.py -backbone "$model" -dataset "$dataset_list"
done
