# MUSO
The code is the implementation of Machine Unlearning by Setting Optimal labels (MUSO), as described in the paper ["MUSO: Achieving Exact Machine Unlearning in Over-Parameterized Regimes"](https://arxiv.org/abs/2410.08557).

For the over-parameterized linear model, enter the folder ***"Linear"***, and run "RunMain.py".

For the neural network model, enter the folder ***"NN"***, and our code's directory structure is as follows:
```
.
|-- data
|-- full_and_sub_class
`-- random_subset
```
***"data"*** includes the nessenary dataset. 
***"full_and_sub_class"*** includes the code for full-class and sub-class unlearning.
***"random_subset"*** includes the code for random-subset unlearning.

### 1. Prepare running environment

### 2. Prepare data

Donwload [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz), [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz) and place the unzipped data in the folder ***"data"***, the resulting directory structure tree is like:
```
data/
|-- cifar10
|   |-- batches.meta
|   |-- data_batch_1
|   |-- data_batch_2
|   |-- data_batch_3
|   |-- data_batch_4
|   |-- data_batch_5
|   |-- readme.html
|   `-- test_batch
`-- cifar100
    |-- file.txt~
    |-- meta
    |-- test
    `-- train
```

### 3. Unlearning

Enter the folder ***"random_subset"*** or ***"full_and_sub_class"***.

Use the script ***"get_original_model.sh"*** in folder ***"scripts"*** to train original models for different datasets: 
```
bash scripts/get_original_model.sh
```

The resulting checkpoint is saved in folder ***"ckpt/MODEL/Vanilla"***. Then, set the parameter resumeCKPT args in ***"utils/unlearning_util.py"***: line 31, line 33, and line 39. Replace the path before ***"model-{mode}.pth"*** with corresponding path.


To retrain the model, run the script ***"retrain_model.sh"*** in folder ***"scripts"***:
```
bash scripts/retrain_model.sh
```

In full-class and sub-class unlearning, we test "rocket", "sea", and "cattle".


