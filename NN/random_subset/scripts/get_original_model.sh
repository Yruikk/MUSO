#!/bin/bash

python train_origin.py --model=ResNet18 --dataset=cifar100 --opt=sgd --lr=0.1 --lr_scheduler=step --wd=5e-4 --batchsize=128 -nd 
