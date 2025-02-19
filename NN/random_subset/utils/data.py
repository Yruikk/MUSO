### 数据集获取
import os

import torchvision
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import torch
import time, random
from torchtoolbox.transform import Cutout

def unpickle(filename):
    with open(filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


transform_train_tinyimagenet = transforms.Compose([
    transforms.RandomCrop(64, padding=8),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()  # 会将数据归一化0-1
])
transform_valid_tinyimagenet = transforms.Compose([
    transforms.ToTensor()  # 会将数据归一化0-1
])

transform_cutout_tinyimagenet = transforms.Compose([
    Cutout(),
    transforms.ToTensor()  # 会将数据归一化0-1
])

transform_train_cifar = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()  # 会将数据归一化0-1
])

transform_train_cifar_patch = transforms.Compose([
    transforms.RandomCrop(32, padding=2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()  # 会将数据归一化0-1
])
transform_valid_cifar = transforms.Compose([
    transforms.ToTensor()  # 会将数据归一化0-1
])

def get_random_subset_idx(dataset_name, forget_per):
    if dataset_name == "cifar10":
        random_subset_idx = np.load(f"idx/cifar10_idx_{forget_per}%.npy")
    elif dataset_name == "cifar100":
        random_subset_idx = np.load(f"idx/cifar100_idx_{forget_per}%.npy")
    elif dataset_name == "tinyimagenet":
        random_subset_idx = np.load(f"idx/tinyimagenet_idx_{forget_per}%.npy")
    else:
        raise NotImplementedError(f"Not implement this dataset : {dataset_name}")
    print("Random Subset Idx:", random_subset_idx[:10])
    return random_subset_idx


class DatasetForRandomSubsetUnlearning(Dataset):
    def __init__(self, transform, forget_idx=np.array([], dtype=np.int64), isForgetSet=False):
        self.forget_idx = forget_idx
        self.isForgetSet = isForgetSet
        self.transform = transform
        
    def __getitem__(self, index):
        tmp_data = self.data[index]
        img = Image.fromarray(np.transpose(tmp_data, (1, 2, 0)))
        return self.transform(img), self.label[index]

    def __len__(self):
        return self.data.shape[0]
    
    def process_data(self):
        if not self.isForgetSet:
            self.data = np.delete(self.data, self.forget_idx, axis=0).copy()
            self.label = np.delete(self.label, self.forget_idx, axis=0).copy()
        else:
            self.data = self.data[self.forget_idx].copy()
            self.label = self.label[self.forget_idx].copy()
    

class DatasetTinyImageNet(DatasetForRandomSubsetUnlearning):
    def __init__(self, mode, transform, forget_idx=np.array([], dtype=np.int64), isForgetSet=False, ):
        if mode == "train":
            self.data = np.load("../data/tiny-imagenet-200/train_data.npy")
            self.label = np.load("../data/tiny-imagenet-200/train_label.npy")
        elif mode == "test":
            self.data = np.load("../data/tiny-imagenet-200/val_data.npy")
            self.label = np.load("../data/tiny-imagenet-200/val_label.npy")
        else:
            raise KeyError("TinyImageNet Dataset mode error")        
        super().__init__(transform=transform, forget_idx=forget_idx, isForgetSet=isForgetSet)
        self.process_data()
    
class DatasetCifar10(DatasetForRandomSubsetUnlearning):
    def __init__(self, mode, transform, forget_idx=np.array([], dtype=np.int64), isForgetSet=False, ):
        if mode == "train":
            datas_with_label = [unpickle('../data/cifar10/data_batch_{}'.format(i + 1)) for i in range(5)]
            self.data = np.concatenate([i[b'data'] for i in datas_with_label], axis=0).reshape(50000, 3, 32, 32)
            self.label = [label for labels in [data[b'labels'] for data in datas_with_label] for label in labels]
            self.label = np.array(self.label)
        elif mode == "test":
            import pickle
            with open('../data/cifar10/test_batch', 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            self.data = dict[b'data'].reshape(10000, 3, 32, 32)
            self.label = np.array(dict[b'labels'])
        else:
            raise KeyError("Cifar10 Dataset mode error")     
        super().__init__(transform=transform, forget_idx=forget_idx, isForgetSet=isForgetSet)
        self.process_data()   
        
class DatasetCifar100(DatasetForRandomSubsetUnlearning):
    def __init__(self, mode, transform, forget_idx=np.array([], dtype=np.int64), isForgetSet=False, ):
        if mode == "train":
            dict = unpickle('../data/cifar100/train')
        elif mode == "test":
            dict = unpickle('../data/cifar100/test')
        else:
            raise KeyError("Cifar100 Dataset mode error")    
        self.data = dict[b'data'].reshape(-1, 3, 32, 32)
        self.label = np.array(dict[b'fine_labels'])    
        super().__init__(transform=transform, forget_idx=forget_idx, isForgetSet=isForgetSet)
        self.process_data()   


class ConcatDataset(Dataset):
    def __init__(self, data1, label1, data2, label2, transform1, transform2):
        self.data1  = data1.copy()
        self.label1 = label1.copy()
        self.data2  = data2.copy()
        self.label2 = label2.copy()
        
        self.transform1 = transform1
        self.transform2 = transform2
        
    def __getitem__(self, index):
        if index < self.data1.shape[0]:   
            tmp_data = self.data1[index]
            img = Image.fromarray(np.transpose(tmp_data, (1, 2, 0)))
            return self.transform1(img), self.label1[index], True
        
        index = index - self.data1.shape[0]
        tmp_data = self.data2[index]
        img = Image.fromarray(np.transpose(tmp_data, (1, 2, 0)))
        return self.transform2(img), self.label2[index], False

    def __len__(self):
        return self.data1.shape[0] + self.data2.shape[0]
    
    

if __name__ == "__main__":
    
    cifar10 = DatasetCifar10("train", transform=transform_train_cifar)
    forget_idx_1 = []
    forget_idx_10 = []
    for i in range(10):
        idxs = np.nonzero(cifar10.label==i)[0]
        np.random.shuffle(idxs)
        forget_idx_1.append(idxs[:50])
        forget_idx_10.append(idxs[:500])
    forget_idx_1 =  np.concatenate(forget_idx_1, axis=0)
    forget_idx_10 =  np.concatenate(forget_idx_10, axis=0)
    
    for i in range(10):
        print(f"class:{i}, num 1:{(cifar10.label[forget_idx_1] == i).sum()}, num 2:{(cifar10.label[forget_idx_10] == i).sum()}")
        
    
    np.save("idx/cifar10_idx_1%.npy", forget_idx_1)
    np.save("idx/cifar10_idx_10%.npy", forget_idx_10)
    
    cifar100 = DatasetCifar100("train", transform=transform_train_cifar)
    forget_idx_1 = []
    forget_idx_10 = []
    for i in range(100):
        idxs = np.nonzero(cifar100.label==i)[0]
        np.random.shuffle(idxs)
        forget_idx_1.append(idxs[:5])
        forget_idx_10.append(idxs[:50])
    forget_idx_1 =  np.concatenate(forget_idx_1, axis=0)
    forget_idx_10 =  np.concatenate(forget_idx_10, axis=0)
    
    for i in range(100):
        print(f"class:{i}, num 1:{(cifar100.label[forget_idx_1] == i).sum()}, num 2:{(cifar100.label[forget_idx_10] == i).sum()}")
    
    np.save("idx/cifar100_idx_1%.npy", forget_idx_1)
    np.save("idx/cifar100_idx_10%.npy", forget_idx_10)
    
    tinyimagenet = DatasetTinyImageNet("train", transform=transform_train_cifar)
    forget_idx_1 = []
    forget_idx_10 = []
    for i in range(200):
        idxs = np.nonzero(tinyimagenet.label==i)[0]
        np.random.shuffle(idxs)
        forget_idx_1.append(idxs[:5])
        forget_idx_10.append(idxs[:50])
    forget_idx_1 =  np.concatenate(forget_idx_1, axis=0)
    forget_idx_10 =  np.concatenate(forget_idx_10, axis=0)
    
    for i in range(200):
        print(f"class:{i}, num 1:{(tinyimagenet.label[forget_idx_1] == i).sum()}, num 2:{(tinyimagenet.label[forget_idx_10] == i).sum()}")
    
    np.save("idx/tinyimagenet_idx_1%.npy", forget_idx_1)
    np.save("idx/tinyimagenet_idx_10%.npy", forget_idx_10)