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

best_model = None
last_model = None


def get_args():
    parser = argparse.ArgumentParser(description='Ours training')
    parser.add_argument('--model', type=str, default='ResNet18')

    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--warmup_epoch', type=float, default=0)
    # parser.add_argument('--adjust_gap', type=int, default=3)
    parser.add_argument('--subset_ratio', type=float, default=0.2)
    parser.add_argument('--epsilon_fix', type=float, default=1e-6)
    parser.add_argument('--epsilon_rs', type=float, default=1e-6)

    parser.add_argument('--opt', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=5.55e-5)  # search
    parser.add_argument('--lr_scheduler', type=str, default="cosine")
    parser.add_argument('--wd', type=float, default=0)

    parser.add_argument('--resume', '-r', action='store_true')  # use -r
    parser.add_argument('--debug', '-nd', action='store_false')  # use -nd, files in /ckpt, else files in /degub_results
    parser.add_argument('--ckpt', type=str, default='ours.pth')
    parser.add_argument('--resumeCKPT', type=str)

    parser.add_argument('--save_path', '-s', action='store_true')
    parser.add_argument('--gpuid', type=str, default='1')

    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--dataset', type=str, default='cifar20')
    parser.add_argument('--forget_class', type=str, default='cattle')
    parser.add_argument('--temp', type=float, default=1, help="KL temperature for ours")
    args = parser.parse_args()
    return args


def get_dataloader(args, num_of_classes):
    np.random.seed(42)

    dataSet, transform_train, transform_valid, forget_class_int = get_dataset_transforms(args)

    retainDS_inTrainSet = dataSet(mode="train", transform=transform_train, forget_class=forget_class_int,
                                  isForgetSet=False, )
    retainDS_inTrainSet_woDataAugment = dataSet("train", transform=transform_valid, forget_class=forget_class_int,
                                                isForgetSet=False)
    forgetDS_inTrainSet_woDataAugment = dataSet("train", transform=transform_valid, forget_class=forget_class_int,
                                                isForgetSet=True)
    retainDS_inTestSet = dataSet("test", transform=transform_valid, forget_class=forget_class_int, isForgetSet=False)
    forgetDS_inTestSet = dataSet("test", transform=transform_valid, forget_class=forget_class_int, isForgetSet=True)
    testDS = dataSet("test", transform=transform_valid)

    trainDS_full = ConcatDataset(
        retainDS_inTrainSet_woDataAugment.data,
        retainDS_inTrainSet_woDataAugment.label,
        forgetDS_inTrainSet_woDataAugment.data,
        forgetDS_inTrainSet_woDataAugment.label,
        transform_valid,
        transform_valid)
    print(f"Data 1  Shape:{trainDS_full.data1.shape}\tData 2 Shape:{trainDS_full.data2.shape}")

    trainDL_full = DataLoader(trainDS_full, batch_size=args.batchsize, shuffle=True, num_workers=8, pin_memory=True)
    retainDL_inTrainSet_woDataAugment = DataLoader(retainDS_inTrainSet_woDataAugment, batch_size=args.batchsize,
                                                   shuffle=False, num_workers=8, pin_memory=True)
    forgetDL_inTrainSet_woDataAugment = DataLoader(forgetDS_inTrainSet_woDataAugment, batch_size=args.batchsize,
                                                   shuffle=False, num_workers=8, pin_memory=True)
    retainDL_inTestSet = DataLoader(retainDS_inTestSet, batch_size=args.batchsize, shuffle=False, num_workers=8,
                                    pin_memory=True)
    forgetDL_inTestSet = DataLoader(forgetDS_inTestSet, batch_size=args.batchsize, shuffle=False, num_workers=8,
                                    pin_memory=True)
    testDL = DataLoader(testDS, batch_size=args.batchsize, shuffle=False, num_workers=8, pin_memory=True)

    return trainDL_full, retainDL_inTrainSet_woDataAugment, forgetDL_inTrainSet_woDataAugment, retainDL_inTestSet, forgetDL_inTestSet, testDL


@torch.no_grad()
def fix_w0_wp(trainDL, args, init_model, pretrain_model, current_model, device):
    init_model.eval()
    pretrain_model.eval()
    current_model.eval()
    for batch_idx, batch_data in enumerate(trainDL):
        inputs = batch_data[0]
        inputs = inputs.to(device)
        outputs_y0 = init_model(inputs)
        outputs_yp = pretrain_model(inputs)
        feature_c = current_model.takefeature(inputs)
        if batch_idx == 0:
            y_0 = outputs_y0
            y_p = outputs_yp
            Z_c = feature_c
        else:
            y_0 = torch.cat([y_0, outputs_y0], dim=0)
            y_p = torch.cat([y_p, outputs_yp], dim=0)
            Z_c = torch.cat([Z_c, feature_c], dim=0)
        # y = N x 10
        # Z = N x 512

    all_ones = torch.ones(Z_c.shape[0], 1)
    all_ones = all_ones.to(device)
    Z_c_wbias = torch.cat([Z_c, all_ones], dim=1)

    eI = args.epsilon_fix * torch.eye(Z_c.shape[1] + 1)
    eI = eI.to(device)

    w_0_fixed = torch.linalg.inv((torch.t(Z_c_wbias) @ Z_c_wbias) + eI) @ torch.t(Z_c_wbias) @ y_0
    w_p_fixed = torch.linalg.inv((torch.t(Z_c_wbias) @ Z_c_wbias) + eI) @ torch.t(Z_c_wbias) @ y_p

    return w_0_fixed, w_p_fixed


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
            total_y = targets
        else:
            retain_data, forget_data = inputs[retain_flag], inputs[~retain_flag]
            retain_targets = targets[retain_flag]
            with torch.no_grad():
                net.eval()
                Z_u = net.takefeature(forget_data)  # N x 512
                all_ones = torch.ones(Z_u.shape[0], 1)
                all_ones = all_ones.to(device)
                Z_u_wbias = torch.cat([Z_u, all_ones], dim=1)  # N x (512+1)

                Z_u_p = pretrain_model.takefeature(forget_data)
                Z_u_p_wbias = torch.cat([Z_u_p, all_ones], dim=1)

                y_u = Z_u_wbias @ right_side - Z_u_p_wbias @ w_p

                _, hard_y_u = torch.max(y_u, dim=1)

                total_x = torch.cat([retain_data, forget_data], dim=0)
                total_y = torch.cat([retain_targets, hard_y_u], dim=0)

                pass
                net.train()

        outputs = net(total_x)
        optimizer.zero_grad()

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
    Z_p_wbias = torch.cat([Z, all_ones], dim=1)  # Z with bias = N x (512+1)

    eI = args.epsilon_rs * torch.eye(Z.shape[0])
    eI = eI.to(device)

    Pi_rp = torch.t(Z_wbias) @ torch.linalg.inv(Z_wbias @ torch.t(Z_wbias) + eI) @ Z_p_wbias  # Pi_r = (512+1) x (512+1)
    Pi_r = torch.t(Z_wbias) @ torch.linalg.inv(Z_wbias @ torch.t(Z_wbias) + eI) @ Z_wbias  # Pi_r = (512+1) x (512+1)
    right_side = Pi_rp @ w_p - Pi_r @ w_0 + w_0 + w_p

    return right_side


if __name__ == '__main__':
    args = get_args()
    # Fix an index fault.
    if args.forget_class == 'cattle':
        args.forget_class = 'vehicle2'
    args.code_file = __file__
    set_resumeCKPT(args)

    args = update_ckpt(args)
    log = LogProcessBar(args.logfile, args)

    # os.environ["CUDA_VISIBLE_DEVICES"] = '1, 3, 0'

    _, gpuid = auto_select_device(memory_max=8000, memory_bias=200, strategy='greedy')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name())
    print(torch.cuda.current_device())

    if args.dataset == 'cifar10':
        num_of_classes = 10
    elif args.dataset == 'cifar100':
        num_of_classes = 100
        init_model1 = torch.load(
            './ckpt/ResNet18/cifar100/Vanilla/24_08_05_21_55_49.20_sgd_lr-0.1_epoch-100_batchsize-128_wd-0.0005/model-init.pth')
        pretrain_model1 = torch.load(
            './ckpt/ResNet18/cifar100/Vanilla/24_08_05_21_55_49.20_sgd_lr-0.1_epoch-100_batchsize-128_wd-0.0005/model-last.pth')
        if args.forget_class == 'rocket':
            retrain_model1 = torch.load(
                './ckpt/ResNet18/cifar100/retrain/rocket/24_08_06_00_21_27.81_sgd_lr-0.1_epoch-100_batchsize-128_wd-0.0005/model-last.pth')
        elif args.forget_class == 'sea':
            retrain_model1 = torch.load(
                './ckpt/ResNet18/cifar100/retrain/sea/24_08_06_01_57_22.01_sgd_lr-0.1_epoch-100_batchsize-128_wd-0.0005/model-last.pth')
    elif args.dataset == 'cifar20':
        num_of_classes = 20
        init_model1 = torch.load(
            './ckpt/ResNet18/cifar20/Vanilla/24_08_05_22_40_04.65_sgd_lr-0.1_epoch-100_batchsize-128_wd-0.0005/model-init.pth')
        pretrain_model1 = torch.load(
            './ckpt/ResNet18/cifar20/Vanilla/24_08_05_22_40_04.65_sgd_lr-0.1_epoch-100_batchsize-128_wd-0.0005/model-last.pth')
        if args.forget_class == 'rocket':
            retrain_model1 = torch.load(
                './ckpt/ResNet18/cifar20/retrain/rocket/24_08_05_23_33_18.60_sgd_lr-0.1_epoch-100_batchsize-128_wd-0.0005/model-last.pth')
        elif args.forget_class == 'sea':
            retrain_model1 = torch.load(
                './ckpt/ResNet18/cifar20/retrain/sea/24_08_06_01_09_25.71_sgd_lr-0.1_epoch-100_batchsize-128_wd-0.0005/model-last.pth')
    elif args.dataset == 'tinyimagenet':
        num_of_classes = 200
    else:
        num_of_classes = 0
        raise NotImplementedError

    w_0_wbias, w_p_wbias, w_r_wbias = get_w_wbias(init_model1, pretrain_model1, retrain_model1)

    print('==> Building model..')
    net = get_model(args.model, num_of_classes=num_of_classes, dataset=args.dataset)
    net = net.to(device)

    if args.resume:
        # Load checkpoint.
        load_model(net, args)
    init_model = deepcopy(net)
    pretrain_model = deepcopy(net)
    retrain_model = deepcopy(net)

    if args.dataset == 'cifar100':
        checkpoint_init = torch.load(
            './ckpt/ResNet18/cifar100/Vanilla/24_08_05_21_55_49.20_sgd_lr-0.1_epoch-100_batchsize-128_wd-0.0005/model-init.pth')
        init_model.load_state_dict(checkpoint_init['net'])
        checkpoint_p = torch.load(
            './ckpt/ResNet18/cifar100/Vanilla/24_08_05_21_55_49.20_sgd_lr-0.1_epoch-100_batchsize-128_wd-0.0005/model-last.pth')
        init_model.load_state_dict(checkpoint_init['net'])
        if args.forget_class == 'rocket':
            checkpoint_r = torch.load(
                './ckpt/ResNet18/cifar100/retrain/rocket/24_08_06_00_21_27.81_sgd_lr-0.1_epoch-100_batchsize-128_wd-0.0005/model-last.pth')
        elif args.forget_class == 'sea':
            checkpoint_r = torch.load(
                './ckpt/ResNet18/cifar100/retrain/sea/24_08_06_01_57_22.01_sgd_lr-0.1_epoch-100_batchsize-128_wd-0.0005//model-last.pth')
        retrain_model.load_state_dict(checkpoint_r['net'])
    elif args.dataset == 'cifar20':
        checkpoint_init = torch.load(
            './ckpt/ResNet18/cifar20/Vanilla/24_08_05_22_40_04.65_sgd_lr-0.1_epoch-100_batchsize-128_wd-0.0005/model-init.pth')
        init_model.load_state_dict(checkpoint_init['net'])
        checkpoint_p = torch.load(
            './ckpt/ResNet18/cifar20/Vanilla/24_08_05_22_40_04.65_sgd_lr-0.1_epoch-100_batchsize-128_wd-0.0005/model-last.pth')
        init_model.load_state_dict(checkpoint_init['net'])
        if args.forget_class == 'rocket':
            checkpoint_r = torch.load(
                './ckpt/ResNet18/cifar20/retrain/rocket/24_08_05_23_33_18.60_sgd_lr-0.1_epoch-100_batchsize-128_wd-0.0005/model-last.pth')
        elif args.forget_class == 'sea':
            checkpoint_r = torch.load(
                './ckpt/ResNet18/cifar20/retrain/sea/24_08_06_01_09_25.71_sgd_lr-0.1_epoch-100_batchsize-128_wd-0.0005/model-last.pth')
        retrain_model.load_state_dict(checkpoint_r['net'])

    if device == 'cuda':
        cudnn.benchmark = True

    tmp_dataloaders = get_dataloader(args, num_of_classes)
    for dataloader in tmp_dataloaders:
        log.log_print(get_dataset_info(dataloader))

    trainDL_full, retainDL_inTrainSet, forgetDL_inTrainSet, retainDL_inTestSet, forgetDL_inTestSet, testDL = tmp_dataloaders

    if args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.opt == 'adamw':
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        optimizer = None
        raise NotImplementedError

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_scheduler(args, len(trainDL_full)), )
    criterion = nn.CrossEntropyLoss()
    epoch_time = AverageMeter("Epoch Time")

    right_side = get_right_side(retainDL_inTrainSet, net, pretrain_model, device, args, w_0_wbias, w_p_wbias)

    for epoch in range(args.epoch):
        start_time = time.time()
        log.log_print("\nEpoch:{} | Model:{} | lr now:{:.6f}".format(epoch, args.model,
                                                                     optimizer.state_dict()['param_groups'][0]['lr']))

        _, _ = train_yrk(net, init_model, pretrain_model, w_p_wbias, trainDL_full, retainDL_inTrainSet, scheduler,
                         optimizer, criterion, log, device, args, right_side)
        _, retain_acc = valid_test(net, retainDL_inTestSet, "Retain Acc on Test", device, criterion, log, args=args)
        _, forget_acc_on_train = valid_test(net, forgetDL_inTrainSet, "Forget Acc on Train", device, criterion, log,
                                            args=args)
        _, forget_acc_on_test = valid_test(net, forgetDL_inTestSet, "Forget Acc on Test", device, criterion, log,
                                           args=args)

        epoch_time.update(time.time() - start_time)
        print("Finished at:" + datetime.datetime.fromtimestamp(
            time.time() + epoch_time.val[-1] * (args.epoch - epoch)).strftime("Time:%H:%M"), )

        right_side = get_right_side(retainDL_inTrainSet, net, pretrain_model, device, args, w_0_wbias, w_p_wbias)

        mia = get_membership_attack_prob(retainDL_inTrainSet, forgetDL_inTrainSet, testDL, net)

    _, test_acc = valid_test(retrain_model, testDL, "Test Acc", device, criterion, log, args=args)
    _, forget_acc = valid_test(retrain_model, forgetDL_inTestSet, "Forget Acc", device, criterion, log, args=args)

    net_w = torch.cat([retrain_model.net.linear.weight, retrain_model.net.linear.bias.reshape(-1, 1)], dim=1)
    net_w_wbias = torch.t(net_w)
    print(torch.linalg.norm(net_w_wbias - w_r_wbias, 'fro') / torch.linalg.norm(w_r_wbias, 'fro'))

    mia = get_membership_attack_prob(retainDL_inTrainSet, forgetDL_inTrainSet, testDL, net)
    log.log_print("Retain Acc:{:.3f}\tForget Acc on Test:{:.3f}\tMIA:{:.3f}".format(
        retain_acc, forget_acc, mia))

    last_model = {
        'net': deepcopy(net.state_dict()),
    }
    saveModel(last_model, args.ckpt.replace('.pth', '-last.pth'))
    saveModel(best_model, args.ckpt.replace('.pth', '-best.pth'))

    save_running_results(args)
