#!/bin/bash

for cls in "rocket" "sea" "cattle"
do
    python retrain.py -r --dataset=cifar20 --forget_class=$cls -nd
    python retrain.py -r --dataset=cifar100 --forget_class=$cls -nd
done
