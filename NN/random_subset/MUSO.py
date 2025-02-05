import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import argparse
from utils import *
import os
import math
import time, datetime
from functools import partial
from copy import deepcopy
from PIL import Image
import matplotlib.pyplot as plt
import wandb

best_model = None
last_model = None


def get_args():
    parser = argparse.ArgumentParser(description='Ours training')
    parser.add_argument('--model', type=str, default='ResNet18')
    parser.add_argument('--method', type=str, default='MUSO')

    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--warmup_epoch', type=float, default=0)
    parser.add_argument('--subset_ratio', type=float, default=0.2)
    parser.add_argument('--epsilon_fix', type=float, default=1e-6)
    parser.add_argument('--epsilon_rs', type=float, default=1e-6)  
    parser.add_argument('--if_fix', type=bool, default=False)

    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--lr_scheduler', type=str, default="constant")
    parser.add_argument('--wd', type=float, default=0)

    parser.add_argument('--resume', '-r', action='store_true')  # use -r
    parser.add_argument('--debug', '-nd', action='store_false')  # use -nd, files in /ckpt, else files in /degub_results
    parser.add_argument('--ckpt', type=str, default='ours.pth')
    parser.add_argument('--resumeCKPT', type=str)

    parser.add_argument('--save_path', '-s', action='store_true')
    parser.add_argument('--gpuid', type=str, default='0')

    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--forget_per', type=int, default=1)
    parser.add_argument('--temp', type=float, default=1, help="KL temperature for ours")
    args = parser.parse_args()
    return args


def get_dataloader(args, num_of_classes):
    np.random.seed(42)
    dataSet, transform_train, transform_valid, random_subset_idx = get_dataset_transform_randomIdx(args)

    retainDS_inTrainSet_woDataAugment = dataSet("train", transform=transform_valid, forget_idx=random_subset_idx,
                                                isForgetSet=False)
    forgetDS_inTrainSet_woDataAugment = dataSet("train", transform=transform_valid, forget_idx=random_subset_idx,
                                                isForgetSet=True)
    testDS = dataSet("test", transform_valid)

    trainDL_full = ConcatDataset(
        retainDS_inTrainSet_woDataAugment.data,
        retainDS_inTrainSet_woDataAugment.label,
        forgetDS_inTrainSet_woDataAugment.data,
        forgetDS_inTrainSet_woDataAugment.label,
        transform_valid,
        transform_valid)

    print(f"Data 1  Shape:{trainDL_full.data1.shape}\tData 2 Shape:{trainDL_full.data2.shape}")
    trainDS_full = DataLoader(trainDL_full, batch_size=args.batchsize, shuffle=True, num_workers=8, pin_memory=True)
    retainDL_inTrainSet_woDataAugment = DataLoader(retainDS_inTrainSet_woDataAugment, batch_size=args.batchsize,
                                                   shuffle=False, num_workers=8, pin_memory=True)
    forgetDL_inTrainSet_woDataAugment = DataLoader(forgetDS_inTrainSet_woDataAugment, batch_size=args.batchsize,
                                                   shuffle=False, num_workers=8, pin_memory=True)
    testDL = DataLoader(testDS, batch_size=args.batchsize, shuffle=False, num_workers=8, pin_memory=True)

    return trainDS_full, retainDL_inTrainSet_woDataAugment, forgetDL_inTrainSet_woDataAugment, testDL


def train_yrk(net, init_model, pretrain_model, w_p, trainDL, retainDL, scheduler, optimizer: optim.Optimizer, criterion,
              log: LogProcessBar, device, args, right_side):
    net.train()
    pretrain_model.eval()
    train_loss = AverageMeter("TrainLoss")
    correct_sample = AverageMeter("CorrectSample")
    total_sample = AverageMeter("TotalSample")
    training_time = AverageMeter("TrainingTime")

    for batch_idx, batch_data in enumerate(trainDL):
        inputs, targets, retain_flag = batch_data[0], batch_data[1], batch_data[2]
        start_time = time.time()
        num_of_batch_samples = inputs.shape[0]
        inputs, targets, retain_flag = inputs.to(device), targets.to(device), retain_flag.to(device)
        retain_flag_int = (retain_flag.reshape(-1, 1)).long()

        if (1 - retain_flag_int).sum() == 0:
            total_x = inputs
            # total_y = pretrain_model(total_x)
            total_y = targets
        else:
            retain_data, forget_data = inputs[retain_flag], inputs[~retain_flag]
            retain_targets = targets[retain_flag]
            # retain_targets = pretrain_model(retain_data)
            with torch.no_grad():
                net.eval()
                Z_u = net.takefeature(forget_data)  # N x 512
                all_ones = torch.ones(Z_u.shape[0], 1)
                all_ones = all_ones.to(device)
                Z_u_wbias = torch.cat([Z_u, all_ones], dim=1)  # N x (512+1)

                Z_u_p = pretrain_model.takefeature(forget_data)
                Z_u_p_wbias = torch.cat([Z_u_p, all_ones], dim=1)

                # y_u = Z_u_wbias @ right_side  # N x 10
                y_u = Z_u_wbias @ right_side - Z_u_p_wbias @ w_p

                _, hard_y_u = torch.max(y_u, dim=1)

                # print(hard_y_u.shape)

                total_x = torch.cat([retain_data, forget_data], dim=0)
                # total_y = torch.cat([retain_targets.float(), y_u.float()])
                total_y = torch.cat([retain_targets, hard_y_u], dim=0)

                pass
                net.train()

        # total_y = F.softmax(total_y / 1, dim=1)
        outputs = net(total_x)
        optimizer.zero_grad()

        # loss = F.kl_div(F.log_softmax(outputs / 1, dim=1), total_y, reduction="batchmean")
        loss = criterion(outputs, total_y)

        loss.backward()

        optimizer.step()
        scheduler.step()

        train_loss.update(loss.item(), num_of_batch_samples)
        correct_sample.update(compute_correct(outputs, targets))
        total_sample.update(num_of_batch_samples)
        training_time.update(time.time() - start_time)

        msg = "[{}/{}] Loss:{} | Acc:{}% | {}".format(
            format_number(2, 3, training_time.avg),
            format_number(3, 2, training_time.sum),
            # datetime.datetime.fromtimestamp(time.time() + (training_time.avg*len(trainloader)+10.6)*(args.epoch -epoch)  ).strftime("Time:%H:%M"),
            format_number(1, 3, train_loss.avg),
            format_number(3, 2, 100. * correct_sample.sum / total_sample.sum),
            "Train".ljust(15),
        )

        if args.debug or (batch_idx == len(trainDL) - 1):
            log.refresh(batch_idx, len(trainDL), msg)


    return 100. * correct_sample.sum / total_sample.sum, right_side


@torch.no_grad()
def get_right_side(retainDL, net, pretrain_model, device, args, w_0, w_p):
    num_subset = int(len(retainDL.dataset) * args.subset_ratio)
    net.eval()
    pretrain_model.eval()
    for batch_idx, batch_data in enumerate(retainDL):
        inputs = batch_data[0]
        inputs = inputs.to(device)
        outputs = net.takefeature(inputs)
        outputs_p = pretrain_model.takefeature(inputs)
        if batch_idx == 0:
            Z = outputs
            Z_p = outputs_p
        else:
            Z = torch.cat([Z, outputs], dim=0)
            Z_p = torch.cat([Z_p, outputs_p], dim=0)

        if Z.shape[0] > 2 * num_subset:
            rand_ind = torch.randperm(Z.size(0))[:num_subset]
            Z = Z[rand_ind]
            Z_p = Z_p[rand_ind]
        # Z = N x 512

    all_ones = torch.ones(Z.shape[0], 1)
    all_ones = all_ones.to(device)
    Z_wbias = torch.cat([Z, all_ones], dim=1)  # Z with bias = N x (512+1)

    eI_N = args.epsilon_rs * torch.eye(Z.shape[0])
    eI_N = eI_N.to(device)

    # Changed by Woodbury Identity - Matrix Invertion Lemma 
    eI_D = args.epsilon_rs * torch.eye(Z.shape[1] + 1)
    eI_D = eI_D.to(device) 

    fast_inv = torch.linalg.inv(eI_N) - torch.linalg.inv(eI_N) @ Z_wbias @ torch.linalg.inv(torch.linalg.inv(eI_D) + torch.t(Z_wbias) @ torch.linalg.inv(eI_N) @ Z_wbias) @ torch.t(Z_wbias) @ torch.linalg.inv(eI_N)
    Pi_r = torch.t(Z_wbias) @ fast_inv @ Z_wbias

    right_side = Pi_r @ (w_p - w_0) + w_0 + w_p

    return right_side


# def main(args):
if __name__ == '__main__':
    args = get_args()
    args.code_file = __file__
    set_resumeCKPT(args)

    args = update_ckpt(args)
    log = LogProcessBar(args.logfile, args)

    # _, gpuid = auto_select_device(memory_max=8000, memory_bias=200, strategy='greedy')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.dataset == 'cifar10':
        num_of_classes = 10
    elif args.dataset == 'cifar100' :
        num_of_classes = 100
    elif args.dataset == 'cifar20':
        num_of_classes = 20
    elif args.dataset == 'tinyimagenet':
        num_of_classes = 200
    else:
        num_of_classes = 0
        raise NotImplementedError

    print('==> Building model..')
    net = get_model(args.model, num_of_classes=num_of_classes, dataset=args.dataset)
    net = net.to(device)

    init_model = torch.load('./ckpt/ResNet18/Vanilla/24_09_06_01_06_26.17_cifar100_sgd_lr-0.1_epoch-100_batchsize-128_wd-0.0005/model-init.pth')
    pretrain_model = torch.load('./ckpt/ResNet18/Vanilla/24_09_06_01_06_26.17_cifar100_sgd_lr-0.1_epoch-100_batchsize-128_wd-0.0005/model-last.pth')

    w_0_wbias, w_p_wbias = get_w_wbias(init_model, pretrain_model)

    if args.resume:
        # Load checkpoint.
        load_model(net, args)
    init_model = deepcopy(net)
    pretrain_model = deepcopy(net)
    retrain_model = deepcopy(net)

    if device == 'cuda':
        cudnn.benchmark = True

    tmp_dataloaders = get_dataloader(args, num_of_classes)
    for dataloader in tmp_dataloaders:
        log.log_print(get_dataset_info(dataloader))

    trainDL, retainDL, forgetDL, testDL = get_dataloader(args, num_of_classes=num_of_classes)

    if args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.opt == 'adamw':
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        optimizer = None
        raise NotImplementedError

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_scheduler(args, len(trainDL)), )
    criterion = nn.CrossEntropyLoss()
    epoch_time = AverageMeter("Epoch Time")

    right_side = get_right_side(retainDL, net, pretrain_model, device, args, w_0_wbias, w_p_wbias)

    angles = np.linspace(0, np.pi, args.epoch)
    rs_values = args.epsilon_rs + (1e-2 - args.epsilon_rs) * (1 - np.cos(angles)) / 2

    for epoch in range(args.epoch):
        start_time = time.time()
        args.epsilon_rs = rs_values[epoch]
        log.log_print("\nEpoch:{} | Model:{} | lr now:{:.6f} | rs now:{:.6f}".format(epoch, args.model,
                                                                    optimizer.state_dict()['param_groups'][0]['lr'], args.epsilon_rs))

        _, _ = train_yrk(net, init_model, pretrain_model, w_p_wbias, trainDL, trainDL, scheduler,
                        optimizer, criterion, log, device, args, right_side)
        _, test_acc = valid_test(net, testDL, "Test Acc", device, criterion, log, args=args)
        _, forget_acc = valid_test(net, forgetDL, "Forget Acc", device, criterion, log, args=args)

        right_side = get_right_side(retainDL, net, pretrain_model, device, args, w_0_wbias, w_p_wbias)

        mia_value = get_membership_attack_prob(retainDL, forgetDL, testDL, net)
        if args.dataset == 'cifar10':
            if args.forget_per == 1:
                avggap = 1/3*(abs(test_acc - 93.29) + abs(forget_acc - 92.96) + abs(mia_value - 79.00))
            elif args.forget_per == 10:
                avggap = 1/3*(abs(test_acc - 92.91) + abs(forget_acc - 93.57) + abs(mia_value - 80.75))
        elif args.dataset == 'cifar100':
            if args.forget_per == 1:
                avggap = 1/3*(abs(test_acc - 76.41) + abs(forget_acc - 75.28) + abs(mia_value - 55.44))
            elif args.forget_per == 10:
                avggap = 1/3*(abs(test_acc - 75.44) + abs(forget_acc - 74.68) + abs(mia_value - 54.59))

        log.log_print("Test Acc:{:.3f}\tForget Acc:{:.3f}\tMIA:{:.3f}\tAvgGap:{:.3f}".format(test_acc, forget_acc, mia_value, avggap))


    last_model = {
        'net': deepcopy(net.state_dict()),
    }
    saveModel(last_model, args.ckpt.replace('.pth', '-last.pth'))
    saveModel(best_model, args.ckpt.replace('.pth', '-best.pth'))

    save_running_results(args)

    
