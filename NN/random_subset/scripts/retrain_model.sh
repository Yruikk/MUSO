#!/bin/bash


    for per in 1 10
    do
        python retrain.py --seed=$seed  --model=ResNet18 --dataset=cifar10 --opt=sgd --lr=0.1 --lr_scheduler=step --wd=5e-4 --batchsize=128 -nd --forget_per=$per
    done


